"""Administration of test invitations (run inside the API container).

    docker compose exec api python -m app.invites create --label "Alice"
    docker compose exec api python -m app.invites list
    docker compose exec api python -m app.invites revoke <id>
    docker compose exec api python -m app.invites add-quota <id> 10
    docker compose exec api python -m app.invites feedback [--csv]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

from app import access_control, paper_registry


def _link(code: str, base_url: str | None) -> str:
    base = (base_url or os.getenv("PUBLIC_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
    # The code is in the URL fragment: browsers never send it to servers, proxies or logs.
    return f"{base}/ui/#invite={code}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.invites", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create an invitation")
    create.add_argument("--label", required=True, help="who it is for (shown in lists and feedback)")
    create.add_argument("--quota", type=int, default=access_control.DEFAULT_QUOTA, help="answered questions (default 20)")
    create.add_argument("--max-attempts", type=int, help="hard cap on questions asked, abstentions included (default 3 x quota)")
    create.add_argument("--max-papers", type=int, default=access_control.DEFAULT_MANUAL_PAPERS, help="manual arXiv additions (default 3)")
    create.add_argument("--days", type=int, help="expire after this many days")
    create.add_argument("--base-url", help="public URL for the link (default: PUBLIC_BASE_URL)")
    commands.add_parser("list", help="list invitations and their usage")
    revoke = commands.add_parser("revoke", help="revoke an invitation and end its sessions")
    revoke.add_argument("id")
    quota = commands.add_parser("add-quota", help="grant extra answered questions")
    quota.add_argument("id")
    quota.add_argument("extra", type=int)
    feedback = commands.add_parser("feedback", help="print the feedback left by testers")
    feedback.add_argument("--csv", action="store_true")
    args = parser.parse_args(argv)

    paper_registry.init_db()
    if args.command == "create":
        invite, code = access_control.create_invite(
            args.label, quota=args.quota, max_attempts=args.max_attempts,
            max_manual_papers=args.max_papers, days=args.days,
        )
        print(f"Invitation {invite.id} for {invite.label!r}: {invite.quota} answered questions"
              f"{', expires ' + invite.expires_at[:10] if invite.expires_at else ''}.")
        print("Link (shown once, only a hash is stored):")
        print(_link(code, args.base_url))
    elif args.command == "list":
        print(f"{'id':10}{'label':24}{'used':>8}{'attempts':>10}{'papers':>8}  status")
        for invite in access_control.list_invites():
            status = "revoked" if invite.revoked else "expired" if invite.expired else "active"
            print(f"{invite.id:10}{invite.label[:22]:24}{invite.used:>3}/{invite.quota:<4}"
                  f"{invite.attempts:>4}/{invite.max_attempts:<5}{invite.manual_papers_used:>3}/{invite.max_manual_papers:<4} {status}")
    elif args.command == "revoke":
        print("revoked" if access_control.revoke_invite(args.id) else "unknown invitation")
    elif args.command == "add-quota":
        print("updated" if access_control.add_quota(args.id, args.extra) else "unknown invitation")
    elif args.command == "feedback":
        rows = access_control.list_feedback()
        if args.csv:
            writer = csv.DictWriter(sys.stdout, fieldnames=["created_at", "label", "rating", "comment", "question", "answer", "conversation_id"])
            writer.writeheader()
            writer.writerows(rows)
        else:
            for row in rows:
                print(f"[{row['created_at'][:16]}] {row['label'] or '-'} {'+' if row['rating'] == 'up' else '-'} "
                      f"{row['question'] or ''}\n    {row['comment'] or ''}")
            print(f"{len(rows)} feedback(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
