"""Room names from what Go2 sees in them (#35).

Every sighting of a telling object is a vote for a room type: a bed for a bedroom, a fridge or an oven for a
kitchen, a couch or a TV for a living room, a toilet for a bathroom, a desk for an office. Objects found in more
than one kind of room (a sink, a dining table) give a smaller vote to each. A room is named once its leading
type has enough votes and clearly beats the next one, so one wrong detection can't name or rename a room. A
name the user gives ("this is the office") always wins over a guess.

Rooms are whatever IDs the room splitting gives them (room regions from #14), so naming doesn't depend on how
rooms are found, and the tests can score it against the houses from #17 on their own.
"""

import threading

# detector label -> {room type: vote per sighting}. Room types are the houses' (simulation/house.py).
EVIDENCE = {
    "bed": {"bedroom": 1.},
    "toilet": {"bathroom": 1.},
    "refrigerator": {"kitchen": 1.},
    "oven": {"kitchen": 1.},
    "microwave": {"kitchen": .5},
    "sink": {"kitchen": .5, "bathroom": .5},
    "dining table": {"kitchen": .5, "living_room": .25},
    "couch": {"living_room": 1.},
    "tv": {"living_room": 1.},
    "desk": {"office": 1.},
}
MIN_VOTES = 2.     # the leading type needs this many votes: two sightings of a bed, or a fridge and an oven
LEAD = 2.          # ... and at least twice the votes of the next type


def tells_room(label):
    """Does a sighting of `label` say anything about which kind of room it's in?"""
    return str(label).strip().lower() in EVIDENCE


class RoomNamer:
    """Collect sightings per room and name each room by its likeliest type."""

    def __init__(self):
        self.votes = {}            # room -> {room type: votes}
        self.given = {}            # room -> the user's name for it
        self._lock = threading.Lock()

    def saw(self, room, label):
        """Count one sighting of `label` in `room`. Labels that say nothing about the room are ignored."""
        evidence = EVIDENCE.get(str(label).strip().lower())
        if room is None or evidence is None:
            return
        with self._lock:
            votes = self.votes.setdefault(room, {})
            for kind, weight in evidence.items():
                votes[kind] = votes.get(kind, 0.) + weight

    def name_room(self, room, name):
        """The user's name for a room, which wins over any guess."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("room name must be nonempty")
        with self._lock:
            self.given[room] = name.strip()

    def room_type(self, room):
        """The room's likeliest type ("bedroom"), or None until the sightings are clear enough."""
        with self._lock:
            return self._type(room)

    def _type(self, room):
        ranked = sorted(self.votes.get(room, {}).items(), key=lambda item: -item[1])
        if not ranked or ranked[0][1] < MIN_VOTES:
            return None
        if len(ranked) > 1 and ranked[0][1] < LEAD * ranked[1][1]:
            return None
        return ranked[0][0]

    def names(self):
        """{room: name} for every named room. Several rooms of one type are numbered in room order
        ("bedroom_1", "bedroom_2"); a single one is just "bedroom"."""
        with self._lock:
            types = {room: self._type(room) for room in self.votes if room not in self.given}
            out = dict(self.given)
            for kind in sorted({t for t in types.values() if t}):
                rooms = sorted(room for room, t in types.items() if t == kind)
                for i, room in enumerate(rooms, 1):
                    out[room] = kind if len(rooms) == 1 else f"{kind}_{i}"
            return out

    def name(self, room):
        """The room's name, or None until it can be named."""
        return self.names().get(room)
