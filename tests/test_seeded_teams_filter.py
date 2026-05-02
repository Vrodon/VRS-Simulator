"""
Tests for participant-list filtering in `_parse_seeded_teams`:

* Teams flagged as withdrawn in `div.inotes-inner` notes are excluded.
* `Stage N Invites` sub-headings under Participants partition the roster
  per stage (used by Major-tier events like IEM Cologne where the main
  page lists 32 teams across 3 invite groups).
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from data_loaders.liquipedia_loader import _parse_seeded_teams


def test_team_flagged_as_withdrawn_is_excluded_from_seeded_list():
    """A team mentioned in an `inotes-inner` 'withdraw from the event' note
    should not appear in the seeded list, even if its teamcard renders."""
    html = """
    <div class="teamcard"><center><a href="/counterstrike/Vitality" title="Vitality">V</a></center></div>
    <div class="teamcard"><center><a href="/counterstrike/FUT_Esports" title="FUT Esports">F</a></center></div>
    <div class="teamcard"><center><a href="/counterstrike/K27" title="K27">K</a></center></div>
    <div class="inotes-inner">
      <dl><dd><small>April 4th - FUT Esports withdraw from the event; they are replaced by K27. [1]</small></dd></dl>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")

    seeded = _parse_seeded_teams(soup)

    assert "FUT" not in seeded            # canonical-normalised form
    assert "FUT Esports" not in seeded
    assert "Vitality" in seeded
    assert "K27" in seeded


def test_parse_per_stage_invites_partitions_teamcards_by_h3():
    """
    Major-tier events (e.g. IEM Cologne) partition the Participants section
    with h3 sub-headings ("Stage 3 Invites", "Stage 2 Invites", "Stage 1
    Invites"). The per-stage parser groups the teamcards under each
    sub-heading so the stage graph can pin them as that stage's roster.
    """
    from data_loaders.liquipedia_loader import _parse_per_stage_invites

    html = """
    <div class="mw-heading mw-heading2"><h2>Participants</h2></div>
    <div class="mw-heading mw-heading3"><h3>Stage 3 Invites</h3></div>
    <div class="teamcard"><center><a href="/counterstrike/Vitality" title="Vitality">V</a></center></div>
    <div class="teamcard"><center><a href="/counterstrike/Spirit" title="Team Spirit">S</a></center></div>
    <div class="mw-heading mw-heading3"><h3>Stage 2 Invites</h3></div>
    <div class="teamcard"><center><a href="/counterstrike/Liquid" title="Team Liquid">L</a></center></div>
    <div class="mw-heading mw-heading3"><h3>Stage 1 Invites</h3></div>
    <div class="teamcard"><center><a href="/counterstrike/FlyQuest" title="FlyQuest">F</a></center></div>
    <div class="mw-heading mw-heading2"><h2>Results</h2></div>
    """
    soup = BeautifulSoup(html, "html.parser")

    invites = _parse_per_stage_invites(soup)

    assert invites == {
        "Stage 1": ["FlyQuest"],
        "Stage 2": ["Liquid"],
        "Stage 3": ["Vitality", "Spirit"],
    }
