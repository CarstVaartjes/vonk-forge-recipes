"""Hardware sweep: qualify as many recipes as possible on the real Sparks.

The package drives a fleet only through ``vonkctl`` (JSON mode). ``policy`` holds
every scheduling decision as a pure function; ``prefetch`` and ``run`` apply
those decisions to the fleet; ``smoke`` exercises a serving recipe; ``state``
and ``report`` keep the resumable record. ``tools/sweep-recipes`` is the entry
point and ``docs/hardware-sweep.md`` explains the design.
"""
