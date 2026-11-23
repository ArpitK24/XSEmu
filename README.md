# XSEmu

XSEmu is an experimental Windows x64 compatibility layer for Xbox Series X|S
software, based on [XCL](https://github.com/ivall/xcl) by ivall and its contributors.
It runs compatible title code on the host CPU and translates Xbox-specific services
and graphics calls. It does not emulate the whole console.

The baseline is [XCL at `f311ca5`](https://github.com/ivall/xcl/tree/f311ca5b969ce2e7303495ccd12c1eed74668383).
It includes the native PE loader, Xbox runtime shims, D3D12 and RDNA shader translation,
virtual title storage, input, audio, and test fixtures. The executable is still
named `xcl.exe`. The [upstream README](UPSTREAM.md) records XCL's reported game
results and limitations; those results are not XSEmu compatibility claims.

## Build and test

Use Windows 10 or 11 x64 with the Visual Studio C++ build tools, the Windows SDK,
and the Rust MSVC toolchain. See the [upstream requirements](UPSTREAM.md#building-and-usage)
for DXC, Gaming Services, and GPU requirements.

```powershell
cargo build --locked --workspace
$env:XCL_ADAPTER = 'warp'
cargo test --locked --workspace -- --test-threads=4 --show-output
```

The workspace builds its own test title. CI runs without retail game files or
the console SDK. Tests that require a physical GPU or optional local samples
may skip on CI. For hardware testing, clear `XCL_ADAPTER` or select your adapter.

To inspect a supported, authorized, already unpacked title before launching it:

```powershell
.\target\debug\xcl.exe inspect D:\Games\Title\game.exe
.\target\debug\xcl.exe run D:\Games\Title
```

Only run titles you trust. The loader runs game code natively, and its virtual
filesystem is not a security sandbox. XSEmu does not decrypt packages or bypass
DRM. Do not commit SDK binaries, game files, firmware, keys, or generated game data.

## Upstream and licensing

The imported source and fixtures retain their existing notices. See [GPL-2.0-only](LICENSE)
and [third-party notices](THIRD_PARTY.md). [OpenScarlett](https://github.com/californiaAlmonds/OpenScarlett),
[XValkyrie](https://github.com/sammwyy/xvalkyrie), [AnyXboxX](https://github.com/LooperSalty/AnyXboxX),
and [Xbox2Windows](https://github.com/R0Y00/Xbox2Windows) informed the foundation
comparison. No additional source from those projects was imported.

Keep one loader and one D3D12 backend. Add missing behavior when a small
reproducible test identifies a gap. Confirm license compatibility and file
provenance before importing any implementation from another project.
