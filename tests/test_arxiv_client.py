import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from app import arxiv_client

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
 <entry>
  <id>http://arxiv.org/abs/2304.02643v4</id>
  <published>2023-04-05T17:57:18Z</published>
  <title>Segment
   Anything</title>
  <summary>We introduce the Segment Anything (SA) project.</summary>
  <author><name>Alexander Kirillov</name></author>
  <author><name>Eric Mintun</name></author>
  <arxiv:primary_category term="cs.CV"/>
 </entry>
 <entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_1</id><title>Error</title></entry>
</feed>"""


def transport(handler):
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


class ArxivClientTests(unittest.TestCase):
    def setUp(self):
        arxiv_client._search_cache.clear()
        patcher = patch.object(arxiv_client, "MIN_SECONDS_BETWEEN_API_CALLS", 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_parse_feed_skips_error_entries(self):
        papers = arxiv_client.parse_atom_feed(FEED)
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual((paper.arxiv_id, paper.latest_version), ("2304.02643", "v4"))
        self.assertEqual(paper.title, "Segment Anything")
        self.assertEqual(paper.authors, ["Alexander Kirillov", "Eric Mintun"])
        self.assertEqual(paper.primary_category, "cs.CV")
        self.assertEqual(paper.published, "2023-04-05")

    def test_search_builds_query_and_caches(self):
        seen = []

        def handler(request):
            seen.append(dict(request.url.params))
            return httpx.Response(200, text=FEED)

        with patch.object(arxiv_client, "_http_client", transport(handler)):
            first = arxiv_client.search("segment anything", max_results=50, category="cs.CV")
            second = arxiv_client.search("segment anything", max_results=50, category="cs.CV")
        self.assertEqual(len(seen), 1)  # second call served from cache
        self.assertEqual(first, second)
        self.assertEqual(seen[0]["search_query"], "all:segment AND all:anything AND cat:cs.CV")
        self.assertEqual(seen[0]["max_results"], "8")  # clamped

    def test_long_natural_queries_use_and_of_words_then_or_fallback(self):
        seen = []

        def handler(request):
            seen.append(request.url.params["search_query"])
            empty = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
            return httpx.Response(200, text=empty if len(seen) == 1 else FEED)

        with patch.object(arxiv_client, "_http_client", transport(handler)):
            results = arxiv_client.search("A Simple Framework for Contrastive Learning of SimCLR")
        self.assertEqual(len(results), 1)
        self.assertEqual(seen[0], "all:Simple AND all:Framework AND all:Contrastive AND all:Learning AND all:SimCLR")
        self.assertEqual(seen[1], "(all:Simple OR all:Framework OR all:Contrastive OR all:Learning OR all:SimCLR)")

    def test_search_validation(self):
        with self.assertRaises(arxiv_client.ArxivError):
            arxiv_client.search("ab")
        with self.assertRaises(arxiv_client.ArxivError):
            arxiv_client.search("valid query", category="cs.CV; DROP")

    def test_http_error_is_safe(self):
        with patch.object(arxiv_client, "_http_client", transport(lambda r: httpx.Response(503))):
            with self.assertRaises(arxiv_client.ArxivError) as caught:
                arxiv_client.lookup(["2304.02643"])
        self.assertIn("503", str(caught.exception))

    def test_split_id(self):
        self.assertEqual(arxiv_client.split_arxiv_id("2304.02643v4"), ("2304.02643", "v4"))
        self.assertEqual(arxiv_client.split_arxiv_id("2304.02643"), ("2304.02643", None))
        with self.assertRaises(arxiv_client.ArxivError):
            arxiv_client.split_arxiv_id("hep-th/9901001")

    def test_download_pdf_ok_and_guards(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "x.pdf"
            ok = transport(lambda r: httpx.Response(200, content=b"%PDF-1.5 data"))
            with patch.object(arxiv_client, "_http_client", ok):
                digest = arxiv_client.download_pdf("2304.02643", "v1", target, max_megabytes=1)
            self.assertEqual(len(digest), 64)
            self.assertTrue(target.is_file())

            html = transport(lambda r: httpx.Response(200, content=b"<html>nope"))
            target2 = Path(directory) / "y.pdf"
            with patch.object(arxiv_client, "_http_client", html):
                with self.assertRaises(arxiv_client.ArxivError):
                    arxiv_client.download_pdf("2304.02643", "v1", target2, max_megabytes=1)
            self.assertFalse(target2.exists())
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["x.pdf"])  # no .part left

            big = transport(lambda r: httpx.Response(200, content=b"%PDF" + b"0" * (2 * 1024 * 1024)))
            with patch.object(arxiv_client, "_http_client", big):
                with self.assertRaises(arxiv_client.ArxivError) as caught:
                    arxiv_client.download_pdf("2304.02643", "v1", Path(directory) / "z.pdf", max_megabytes=1)
            self.assertIn("larger", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
