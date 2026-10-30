# ns-3 build for this project copy

The pre-existing `ns-3.48/cmake-cache` was generated from the original checkout
at `/home/kshit/5g-project`, and the pre-existing `ns-3.48/build` executable
resolves shared libraries from that checkout. Both are retained as the
baseline. The controller now defaults to the separate, self-contained binary
at `ns-3.48/build-copyfix/scratch/ns3.48-qoe-dataset-default`, whose RPATH
resolves libraries from `ns-3.48/build-copyfix/lib` in this project copy.

Rebuild that local binary from the project root with:

```bash
cmake -S ns-3.48 -B ns-3.48/cmake-cache-copyfix -G Ninja \
  '-DNS3_ENABLED_MODULES=antenna;applications;buildings;config-store;core;flow-monitor;internet;internet-apps;mobility;network;nr;point-to-point;propagation;spectrum;stats;traffic-control' \
  -DNS3_EXAMPLES=OFF \
  -DNS3_TESTS=OFF \
  -DNS3_PYTHON_BINDINGS=OFF \
  -DNS3_OUTPUT_DIRECTORY="$PWD/ns-3.48/build-copyfix"
cmake --build ns-3.48/cmake-cache-copyfix --target scratch_qoe-dataset -j 8
```

Check library resolution with:

```bash
ldd ns-3.48/build-copyfix/scratch/ns3.48-qoe-dataset-default | grep libns3.48
```

The paths should all point under this copy's `ns-3.48/build-copyfix/lib`.
