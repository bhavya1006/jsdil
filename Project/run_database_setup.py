"""Creates the database, its tables and the accounts people sign in with. Safe to run again."""
import os
import secrets

from sqlalchemy import inspect

from access.password_security import hash_password
from config import config
from persistence.database import database_engine, session_factory
from persistence.models import Base, UserAccount
from persistence.repository import fetch_user_account, insert_user_account


def _refuse_earlier_versions():
    inspector = inspect(database_engine)
    existing = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name not in existing:
            continue
        stored = {column["name"]: bool(column["nullable"])
                  for column in inspector.get_columns(table.name)}
        declared = {column.name: bool(column.nullable) for column in table.columns}
        if stored != declared:
            raise SystemExit(f"The table {table.name} in {config.PERMIT_DATABASE_FILE} was made by "
                             f"an earlier version: delete it and {config.CHECKPOINT_DATABASE_FILE} "
                             "and run the setup again.")


def main():
    """Give the system its tables and the staff their accounts."""
    os.makedirs(config.DATABASE_DIRECTORY, exist_ok=True)
    _refuse_earlier_versions()
    Base.metadata.create_all(database_engine)
    print(f"database: {config.PERMIT_DATABASE_FILE}")
    print("  tables: " + ", ".join(sorted(table.name for table in Base.metadata.sorted_tables)))
    created, present = 0, 0
    with session_factory() as session:
        for account in config.SEED_USER_ACCOUNTS:
            if fetch_user_account(session, UserAccount.username == account["username"]):
                present += 1
                continue
            salt = secrets.token_hex(config.PASSWORD_SALT_BYTES)
            insert_user_account(session, {
                "username": account["username"], "display_name": account["display_name"],
                "role": account["role"], "area_group": account["area_group"],
                "password_salt": salt,
                "password_hash": hash_password(account["password"], salt),
                "session_token": None})
            created += 1
            print(f"  account created  {account['username']:<{config.USERNAME_WIDTH}} "
                  f"{account['display_name']:<{config.DISPLAY_NAME_WIDTH}} {account['role']}")
        session.commit()
    print(f"  accounts created: {created}, already present: {present}")


if __name__ == "__main__":
    main()
