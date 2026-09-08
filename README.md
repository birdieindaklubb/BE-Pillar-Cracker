VIBECODED Pocket Edition / Bedrock Edition end pillar cracker

It searches the 32-bit PE/BE world-seed space from End
pillar heights, radii, and crystal-cage observations, then can apply exact
base-End-terrain observations to the resulting candidates.

## What it recovers

PE 1.1.5.0 initializes MT19937 directly with its 32-bit world seed, then does a
forward Fisher-Yates shuffle of the values 0 through 9. At End ring index `i`,
the shuffled shape determines every visible pillar property:

```
feature height = 76 + 3 * shape
radius         = 2 + floor(shape / 3)
caged          = shape is 1 or 2
```

Thus height, radius, and cage state are intersecting constraints on the same
native shape, never substitute RNG calls. A complete set of ten heights has
only 10 possible patterns, but does **not** uniquely determine a 32-bit seed:
a complete scan normally finds about 1,184 matching world seeds. This tool
returns every matching full seed, with signed, hexadecimal, high-16, and
low-16 forms. Another independent world-generation observation is needed to
select one candidate.

## Measuring a height

Enter the pillar's **feature height**, one of `76, 79, ..., 103`.  It is the Y
coordinate used for the crystal/cage layer.  The highest obsidian block is one
block lower (`feature_height - 1`), because PE writes obsidian only below that
coordinate.  Use `--obsidian-tops` if that is what was measured.

The ring order used by `--heights` is fixed and follows the PE 1.1.5 decorator:

| index | center X | center Z |
| ---: | ---: | ---: |
| 0 | 42 | 0 |
| 1 | 33 | 24 |
| 2 | 12 | 39 |
| 3 | -12 | 39 |
| 4 | -33 | 24 |
| 5 | -42 | 0 |
| 6 | -33 | -24 |
| 7 | -12 | -39 |
| 8 | 12 | -39 |
| 9 | 33 | -24 |

## Build

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

OpenMP is used automatically when the installed C++ toolchain supports it.
Without it, the result remains exact but scanning uses one worker. The terrain
filter is always included; it has no sibling checkout or runtime dependency.

## Browser recorder

Open [`web/index.html`](web/index.html) directly in a browser to use the
clickable top-down End diagram. It records height, radius, and cage state at
each fixed PE ring position, prevents impossible combinations, and copies a
command for the executable built above. It is a local static page: it has no
server, network requests, analytics, or access to world saves. If the
executable is built somewhere other than `build\\Release`, adjust that first
path in the copied command.

Ready-to-copy command examples are in
[`examples/pillar-constraints.txt`](examples/pillar-constraints.txt). The
terrain filter's input format is demonstrated by
[`examples/terrain-observations.txt`](examples/terrain-observations.txt) and
[`examples/candidate-seeds.txt`](examples/candidate-seeds.txt).

## Use

First check the executable's MT19937/shuffle path and four full embedded
End-terrain chunk regressions:

```powershell
.\build\Release\pe115_pillarcracker.exe --self-test
```

Use all ten feature heights in ring order, then choose the seed space to scan.
For example, if another observation has narrowed the high 16 bits:

```powershell
.\build\Release\pe115_pillarcracker.exe `
  --heights 76,79,82,85,88,91,94,97,100,103 `
  --high16 0x1234
```

Every scan and `--terrain-filter` run also writes its candidate list to
`results\pe115-YYMMDD-HHMMSS-MMM.txt` beneath the current working directory.
The compact local-time identifier includes milliseconds; if it still collides,
the tool adds a short numeric suffix rather than replacing an existing file.
The file contains the same `unsigned=...` entries printed to the terminal and
can be passed directly to `--terrain-filter`.

For pillars recorded with coordinates rather than ring indices, repeat
`--pillar X,Z,HEIGHT[,RADIUS[,CAGED|UNCAGED]]` in any order. For an observed
highest obsidian block instead, use `--pillar-top X,Z,TOP[,RADIUS[,CAGED|UNCAGED]]`:

```powershell
.\build\Release\pe115_pillarcracker.exe `
  --pillar 42,0,76 --pillar 33,24,79 --pillar 12,39,82 `
  --pillar -12,39,85 --pillar -33,24,88 --pillar -42,0,91 `
  --pillar -33,-24,94 --pillar -12,-39,97 --pillar 12,-39,100 `
  --pillar 33,-24,103 --low16 0xbeef
```

Height is optional when another property is known. Add one or both independent
forms in any combination, including alongside `--pillar`:

```text
--pillar-radius X,Z,RADIUS             RADIUS is 2, 3, 4, or 5
--pillar-cage X,Z,CAGED|UNCAGED        CAGED is accepted as CAGE, YES, TRUE, or 1;
                                       UNCAGED is accepted as NO-CAGE, NO, FALSE, or 0
```

For example, `--pillar-radius 42,0,4 --pillar-cage 42,0,UNCAGED` is valid even
when the height at that ring position is unknown. Contradictory combinations,
such as a caged radius-4 pillar, are rejected before scanning.

Other exact scan selectors are:

```text
--low16 VALUE                       scan every seed with this low word
--range-start VALUE --count VALUE   scan one contiguous 32-bit range
--all                               explicitly scan all 2^32 world seeds
--threads COUNT                     cap OpenMP worker count
--cuda                              require the optional exact CUDA backend
--avx2                              require the optional exact AVX2 backend
--scalar                            force the portable scalar scanner
```

`--all` is deliberately opt-in: an exhaustive 2^32 scan is exact but can take
hours on a normal desktop.  Divide it into non-overlapping `--range-start` /
`--count` slices when running across sessions or machines.

`--threads COUNT` caps the CPU workers used by a scan or by
`--terrain-filter`. The terrain filter otherwise uses the available OpenMP
workers; builds without OpenMP remain exact and run serially.

The default selects an available exact CUDA backend first, then an AVX2
eight-seed CPU implementation, then the portable scalar implementation. CUDA
is enabled only when a CUDA compiler is found while configuring; `--cuda`
requires both that build support and a usable NVIDIA device. `--avx2` likewise
requires a build with AVX2 support and an AVX2-capable CPU. Every backend uses
the same MT19937 and shuffle calculation. `--scalar` is useful for comparison
or older CPUs.

The command-line scanner accepts any non-empty combination of per-pillar
constraints. A fully resolved height permutation uses the optimized
inverse-shuffle predicate. Partial height/radius/cage observations use the
same exact forward shuffle per candidate and can return many seeds; prefer a
high/low-word or range selector when the information is weak. You can inspect
the predicted layout, including radius and cage state, for any known candidate
with:

```text
--verify SEED
```

`SEED` and selector values accept decimal or `0x` hexadecimal notation.

## Exact End-terrain filter

After obtaining a candidate list from the pillar scan, record blocks from the
unmodified **main End island's base terrain** in a text file. Each non-comment
line is:

```text
x y z value
```

`value` is `1` for End stone and `0` for air. Commas may replace spaces; blank
lines and `#` comments are accepted. Coordinates use normal world block
coordinates and `y` must be in `0..127`. For example:

```text
# x  y   z   End stone (1) / air (0)
0   56  0   1
0   127 0   0
```

Use the pillar scanner's saved stdout directly, or provide one signed or
unsigned 32-bit decimal seed (or a `0x` hexadecimal bit pattern) per line:

```powershell
.\build\Release\pe115_pillarcracker.exe `
  --terrain-filter .\pillar-candidates.txt .\end-terrain.txt
```

The command tests every candidate with its embedded PE 1.1.5 End base-chunk
generator and prints only the survivors in the same machine-readable
`unsigned=...` form, so filters can be chained. It compares `1` specifically
to End stone and `0` specifically to air; it does not treat an arbitrary
non-End-stone block as air.

Only record natural terrain before decoration or player changes. Do not record
the arrival platform, exit portal, pillars, entities, mined/placed blocks, or
other non-base-terrain locations: those are not part of this seed-only terrain
predicate. Terrain observations reduce candidates; they do not change or tune
the generator.

## Exactness boundaries

The optimized scan derives only the first nine MT19937 outputs after the
initial twist, because those are the only random values the PE pillar shuffle
consumes.  It is algebraically the same MT initialization, twist, tempering,
modulo reduction, and forward shuffle as the target behavior; it does not use
an approximation or a substitute PRNG.  The optimization is solely avoiding
the unused remainder of MT19937's 624-word state.

This tool supports PE 1.1.5 pillar candidates and its base-End-terrain filter
only. It does not claim that pillars alone recover a unique seed, and does not
support other PE or Java versions.

The embedded terrain path is deliberately narrow and version-pinned. It uses
the native source's 3x3x33 density lattice, 8x4x8 interpolation, 16/16/8
noise-octave construction order, binary32 operation boundaries, and End-stone
threshold. It is not a terrain heuristic or a Java Edition generator.
