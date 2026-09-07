VIBECODED Pocket Edition 1.1.5 end pillar cracker

## What it recovers

PE 1.1.5.0 initializes MT19937 directly with its 32-bit world seed, then does a
forward Fisher-Yates shuffle of the values 0 through 9.  At End ring index
`i`, the generator's feature height is:

```
76 + 3 * shuffled_value[i]
```

Therefore a complete set of ten heights has only 10 possible patterns.  It
does **not** uniquely determine a 32-bit seed: a complete scan normally finds
about ~1,184 matching world seeds.  This tool returns every matching full seed,
with signed, hexadecimal, high-16, and low-16 forms.  Another independent
world-generation observation is needed to narrow it down to one candidate.

`--high16` and `--low16` are filters for information obtained
elsewhere; they do not infer a seed half from the pillars alone.

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
Without it, the result remains exact but scanning uses one worker.

## Browser recorder

Open [`web/index.html`](web/index.html) directly in a browser to use the
clickable top-down End diagram. It records the ten heights in the fixed PE ring
order, prevents duplicate values, and copies a command for the executable built
above. It is a local static page: it has no server, network requests, analytics,
or access to world saves. If the executable is built somewhere other than
`build\\Release`, adjust that first path in the copied command.

## Use

First check that the executable and the embedded MT19937 prefix calculation
are sound:

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

For pillars recorded with coordinates rather than ring indices, repeat
`--pillar X,Z,HEIGHT` in any order:

```powershell
.\build\Release\pe115_pillarcracker.exe `
  --pillar 42,0,76 --pillar 33,24,79 --pillar 12,39,82 `
  --pillar -12,39,85 --pillar -33,24,88 --pillar -42,0,91 `
  --pillar -33,-24,94 --pillar -12,-39,97 --pillar 12,-39,100 `
  --pillar 33,-24,103 --low16 0xbeef
```

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

The default selects an available exact CUDA backend first, then an AVX2
eight-seed CPU implementation, then the portable scalar implementation. CUDA
is enabled only when a CUDA compiler is found while configuring; `--cuda`
requires both that build support and a usable NVIDIA device. `--avx2` likewise
requires a build with AVX2 support and an AVX2-capable CPU. Every backend uses
the same MT19937 and shuffle calculation. `--scalar` is useful for comparison
or older CPUs.

The program deliberately requires all ten unique pillar heights when scanning.
Partial height data admits far too many full seeds to be a useful or safe
default.  You can inspect the predicted layout for any known candidate with:

```text
--verify SEED
```

`SEED` and selector values accept decimal or `0x` hexadecimal notation.

## Exactness boundaries

The optimized scan derives only the first nine MT19937 outputs after the
initial twist, because those are the only random values the PE pillar shuffle
consumes.  It is algebraically the same MT initialization, twist, tempering,
modulo reduction, and forward shuffle as the target behavior; it does not use
an approximation or a substitute PRNG.  The optimization is solely avoiding
the unused remainder of MT19937's 624-word state.