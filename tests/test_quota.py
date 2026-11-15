"""Quota tier tests: which daily limit a member's roles give them."""

import quota

GUILD = "guild-1"
LIMITS = {GUILD: {"apprentice": 20, "ascendant": 150}}


def test_no_listed_role_gets_default(monkeypatch):
    monkeypatch.setattr(quota, "GUILD_ROLE_LIMITS", LIMITS)
    assert quota.daily_limit(GUILD, ["some-other-role"]) == quota.DEFAULT_DAILY_LIMIT


def test_highest_role_wins(monkeypatch):
    monkeypatch.setattr(quota, "GUILD_ROLE_LIMITS", LIMITS)
    assert quota.daily_limit(GUILD, ["apprentice", "ascendant"]) == 150


def test_unknown_guild_and_dm_get_default(monkeypatch):
    monkeypatch.setattr(quota, "GUILD_ROLE_LIMITS", LIMITS)
    assert quota.daily_limit("another-guild", ["ascendant"]) == quota.DEFAULT_DAILY_LIMIT
    assert quota.daily_limit("dm", []) == quota.DEFAULT_DAILY_LIMIT
