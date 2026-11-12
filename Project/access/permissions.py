"""The sign-in and permission guard, and the hold of an Area Authority to its own plant areas."""
from typing import Annotated

from fastapi import Header, HTTPException, Request

from config import config
from persistence.database import session_factory
from persistence.models import UserAccount
from persistence.repository import fetch_user_account

SessionTokenHeader = Annotated[str | None, Header(alias=config.SESSION_TOKEN_HEADER)]


def authorize_request(request: Request, session_token: SessionTokenHeader = None):
    """Who is asking, and whether their role holds the permission the route needs."""
    if not session_token:
        raise HTTPException(status_code=401, detail=config.SIGN_IN_REQUIRED_MESSAGE)
    with session_factory() as session:
        account = fetch_user_account(session, UserAccount.session_token == session_token)
    if account is None:
        raise HTTPException(status_code=401, detail=config.SIGN_IN_REQUIRED_MESSAGE)
    route = request.scope.get("route")
    path_template = getattr(route, "path", request.url.path)
    permission = config.PERMISSION_BY_ROUTE.get((request.method, path_template))
    if permission is None or permission not in config.ROLE_PERMISSIONS.get(account["role"], []):
        raise HTTPException(status_code=403, detail=config.PERMISSION_REFUSED_MESSAGE.format(
            role=account["role"], permission=permission))
    return account


def authorize_plant_area(signed_in_user, plant_area, permit_reference):
    """Refuse an Area Authority acting outside the plant areas of its own area group."""
    if signed_in_user.get("role") != config.ROLE_AREA_AUTHORITY:
        return None
    area_group = signed_in_user.get("area_group")
    if plant_area not in config.AREA_GROUPS.get(area_group, []):
        raise HTTPException(status_code=403, detail=config.AREA_GROUP_REFUSED_MESSAGE.format(
            username=signed_in_user.get("username"), area_group=area_group,
            permit_reference=permit_reference, plant_area=plant_area))
    return None
