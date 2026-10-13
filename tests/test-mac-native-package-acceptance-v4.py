"""Inert Stage B contracts; no child, model, socket or native limit execution."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import os
import stat
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

PARENT = Path(__file__).parent / "native/mac-native-package-acceptance-v4.py"
BOOTSTRAP = Path(__file__).parent / "native/mac-native-package-bootstrap-v4.py"
PACKAGE_RAW = b'{\n  "A_runtime_packet_path": ".work/reports/mac-native-runtime-pins-v1.json",\n  "A_runtime_packet_sha256": "8db57e3f700aebb3adeb383c719a4bc5e2525917f4c017c853ee4694be1ee1e4",\n  "OS": {\n    "kernel_release": "25.6.0",\n    "kernel_version": "Darwin Kernel Version 25.6.0: Tue Aug 18 17:49:54 PDT 2026; root:xnu-12377.161.15.700.19~2/RELEASE_ARM64_T8142",\n    "machine": "arm64"\n  },\n  "all_publication_held": true,\n  "created_utc": "2026-10-12T09:07:42.262934+00:00",\n  "current_wheel_artifact_packet_path": ".work/reports/memory-installed-wheel-artifact-v2.json",\n  "current_wheel_artifact_packet_sha256": "4b82143e4adaa5e26d81bbdfb32fc725d171bf0e466fe7ee0bfe14c1bc0c062f",\n  "data": {\n    "truememory/data/l3_weights.json": "5d91643f843b4639375a4324aa87cbd09815a4242b11f3fe4138bae641b2d5b8"\n  },\n  "data_map_sha256": "19d20af55c695fe0952cfc26e4fd339037ac3bbfe05fab4b14bb1ae814ca4ea2",\n  "interpreter_linkage_packet_path": ".work/reports/mac-native-interpreter-linkage-v1.json",\n  "interpreter_linkage_packet_sha256": "762d535cdb9f021729fe101c654966e52a9d8da285d5fe77bc0b1db815d43843",\n  "interpreter_runtime_packet_path": ".work/reports/mac-native-interpreter-runtime-v1.json",\n  "interpreter_runtime_packet_sha256": "73a77a325ba667ff84d395c8ecddd9d21d63be27914c4586c6d5e65e2ca55596",\n  "live_service_queried": false,\n  "metadata_identity_mode": "Actual current Linux py3-none-any wheel metadata staged privately; not installed Mac package",\n  "model_loads": 0,\n  "modules_imported": false,\n  "native_GO": false,\n  "numpy_MachO_dependency_names": {\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_multiarray_tests.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-darwin.so": [\n      "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",\n      "/usr/lib/libc++.1.dylib",\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_operand_flag_tests.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_rational_tests.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_simd.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_struct_ufunc_tests.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_umath_tests.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/fft/_pocketfft_umath.cpython-312-darwin.so": [\n      "/usr/lib/libc++.1.dylib",\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/linalg/_umath_linalg.cpython-312-darwin.so": [\n      "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/linalg/lapack_lite.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib",\n      "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_bounded_integers.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_common.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_generator.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_mt19937.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_pcg64.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_philox.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_sfc64.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/bit_generator.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ],\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/mtrand.cpython-312-darwin.so": [\n      "/usr/lib/libSystem.B.dylib"\n    ]\n  },\n  "numpy_dependency_root": "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy",\n  "numpy_files": {\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/__config__.py": {\n      "bytes": 4867,\n      "sha256": "062c14b924ddd2c1297088a9a9623607a90ab73c8bc56dd6087ca4bde928e739"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/__init__.py": {\n      "bytes": 26277,\n      "sha256": "92a46f791e453926d3292af2582b89995a289475f0eaaea71a949823200b838a"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_array_api_info.py": {\n      "bytes": 10385,\n      "sha256": "ce6ec789062aca14aa165dc9586ece2e1130f891843eaefeefea968575aa559f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_configtool.py": {\n      "bytes": 1007,\n      "sha256": "105449de96b34f162113da29fa8716c8a4cb66bae91617e6992fed5ababc0b1a"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/__init__.py": {\n      "bytes": 6662,\n      "sha256": "837ce8aec8693095d2e1c7c306d385d1785a50be97a009935c64cae04e3555d2"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_add_newdocs.py": {\n      "bytes": 216086,\n      "sha256": "bdb68c6da2f3a1048cb2d6123afb6ede6f2bee88c88acb4d330c5485dc1324ce"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_add_newdocs_scalars.py": {\n      "bytes": 12933,\n      "sha256": "db6f2b889f9dcfd7d5df64ef3a430f532b3fdbfc279c3b94f8451b1757a16efb"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_asarray.py": {\n      "bytes": 3894,\n      "sha256": "f5aa779032cd51c8ca49039fae454fcebd2d64647513576d0884ff61e68525d2"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_dtype.py": {\n      "bytes": 10547,\n      "sha256": "70ce899e3a072d445608780df15990ca37a28838dcc210792ff7cf30e7b5bae3"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_dtype_ctypes.py": {\n      "bytes": 3726,\n      "sha256": "28f3e56a40ec3e4b938523abe6a1705b48c9f559e36ef5b811684e6c21d81881"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_exceptions.py": {\n      "bytes": 5159,\n      "sha256": "5fc120d61ab5b94f0bf7088ec05a368d1aba4b4be78c27606051e3de1016f42a"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_internal.py": {\n      "bytes": 29437,\n      "sha256": "c15b426aca730fccd5261139ee1a825abaf298d21d2522aa7f37338cd44598ee"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_methods.py": {\n      "bytes": 9393,\n      "sha256": "724facf7e63c5b8fcc2fac14bf31a02cb048d9e28f516d86502ba5077a425476"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_multiarray_tests.cpython-312-darwin.so": {\n      "bytes": 120952,\n      "sha256": "7eebe35f8c33291691312a8418d770da61837330588f966fbae8ad88099631eb"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_multiarray_umath.cpython-312-darwin.so": {\n      "bytes": 3678408,\n      "sha256": "b96123b811488473c23efa6d45f196641463c928372ff896785813b821aed652"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_operand_flag_tests.cpython-312-darwin.so": {\n      "bytes": 51240,\n      "sha256": "349c39c8494d2439e6732a0f8caf105a4737f21cbf1b962e1b07b36d853b58f6"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_rational_tests.cpython-312-darwin.so": {\n      "bytes": 72808,\n      "sha256": "0bfbea6b715e9c44c438b868b76e252178a478ecd094a1521d086165b55fdf22"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_simd.cpython-312-darwin.so": {\n      "bytes": 339592,\n      "sha256": "3178cd1f5ab8625fa8371bf48dbe54b4ed88896f9f70519c34ccaea970374197"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_string_helpers.py": {\n      "bytes": 2845,\n      "sha256": "e929a0a22ea80f60ae9e3c014abf41676d079029ef5ba9d33db953394de95a78"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_struct_ufunc_tests.cpython-312-darwin.so": {\n      "bytes": 51480,\n      "sha256": "9337744b95313d29845b741049a8cdaa8abe57520524e7f16c396b2ae64f1d32"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_type_aliases.py": {\n      "bytes": 3755,\n      "sha256": "a4d4a14f10ba6af2c1d912fbf8f38eb69150469e773bee5480e4f1962cf58296"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_ufunc_config.py": {\n      "bytes": 15611,\n      "sha256": "0e938bb63600619bfe9028e8f285f9ddcd925b62d0220690ec4a650a48ac45a5"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/_umath_tests.cpython-312-darwin.so": {\n      "bytes": 71712,\n      "sha256": "ba6ac034d6e710660862a3260c4e8d62171de92a9b3ea32029bc17774ea7f84f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/arrayprint.py": {\n      "bytes": 65271,\n      "sha256": "f7a7c9a96b764436f675560f2dda3990d0d042ae8aa898d2baa40e7f5918209b"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/cversions.py": {\n      "bytes": 347,\n      "sha256": "1ff88d229c7dfa1635710371aa34f677fe525d98496cca3f71aab8feae8b07b2"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/defchararray.py": {\n      "bytes": 37817,\n      "sha256": "30b125c1f98ae7bea0cf9f80fbb49558867dece970d72e675ff7f0c6c84d22f7"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/einsumfunc.py": {\n      "bytes": 57930,\n      "sha256": "b91287b000d50c68c3be44597f9d968d4e18a967c729143aa5efe012bf4f847d"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py": {\n      "bytes": 142658,\n      "sha256": "6055d8e195cf7aaae417d23bda203cd35014e3db3d7c28a32e7eb71e858e945f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/function_base.py": {\n      "bytes": 19677,\n      "sha256": "b8b8f508410c567b0ea4b44db18e28eb09c62e9ec33680fcefdea7de82eeb9c1"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/getlimits.py": {\n      "bytes": 15003,\n      "sha256": "46d3015808f29584e67eab482f52542a02fb6f0c95c2ca1bd386dc42b9a9ed72"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/memmap.py": {\n      "bytes": 12651,\n      "sha256": "c88b10ea7f64a59ba58204492459136e35709c1e257a8ca2cef529736894e27f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/multiarray.py": {\n      "bytes": 56786,\n      "sha256": "ae9b35e60d0150e2ade05d45c1a31dbfeee627dbe668b2b7a1bfc2e3fe53f77f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/numeric.py": {\n      "bytes": 83038,\n      "sha256": "d75e4d3072e59b477cad20f088d964f6b41c840e7304d3f76fe9f6832d760a9f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/numerictypes.py": {\n      "bytes": 15967,\n      "sha256": "d7924b057d26fcc4206a21ffc81048e6096fd2f5ee794d1aae74b9e915df5319"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/overrides.py": {\n      "bytes": 7480,\n      "sha256": "2bdc18ac0e766e23be366f0605d0f382ea9272d2500e8baf03821e91395fc555"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/printoptions.py": {\n      "bytes": 1056,\n      "sha256": "345a6fcb96e78dbbea9ca7aa42dd2f784131a400180153688865c7de98255807"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/records.py": {\n      "bytes": 36754,\n      "sha256": "fd40158fe835806fa832a449440e04f414f72331b733825fc58dd48e6255330d"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/shape_base.py": {\n      "bytes": 32712,\n      "sha256": "1d6e897dfdb7edf2d71e8ed75254a6e7106b1a947592f88d91af96d1ad2c469e"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/strings.py": {\n      "bytes": 50578,\n      "sha256": "a9c5e0f420035ed0f3a5f51bd95307c0a78f3389d764f84da46a554e35097065"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_core/umath.py": {\n      "bytes": 2140,\n      "sha256": "450a42560a838a2260e85f23c8398b7d09e7a8d8dd4170376e0d920463060c0b"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_distributor_init.py": {\n      "bytes": 421,\n      "sha256": "14148976054795071ae41ad011560fa059ba2924c98481675ad59b1241214d2a"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_expired_attrs_2_0.py": {\n      "bytes": 3771,\n      "sha256": "735b4637a636441b0324e9498dcff127231ea3a1d321d939c75048c6dfab2684"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_globals.py": {\n      "bytes": 4155,\n      "sha256": "d37b1eb1df7d7ae55fdc2bd9d757631344be0bacfe56d8456062b60b23e3722b"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_pyinstaller/__init__.py": {\n      "bytes": 0,\n      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_pyinstaller/hook-numpy.py": {\n      "bytes": 1362,\n      "sha256": "314db6a50e0091460f416bb90bca510e99d85c494b27c4741463582544298951"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_pytesttester.py": {\n      "bytes": 6328,\n      "sha256": "0e39582fcb8834dd9759ade79e55fa80f1aea0b8dcc750627bf3d0cdba767290"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/__init__.py": {\n      "bytes": 5466,\n      "sha256": "33034ac192a99a0eecf9babd14db781811274926d04a6923f92a71b66329a229"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_add_docstring.py": {\n      "bytes": 3999,\n      "sha256": "ff783b0fee87010dcc4f85fa0c4d3bc8bb9af4ba96161b24355c754d2ed7f4ee"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_array_like.py": {\n      "bytes": 4188,\n      "sha256": "10f6547c94a36a6bec589e91b39670c00ff916106961da227dc555b557163e7d"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_char_codes.py": {\n      "bytes": 8764,\n      "sha256": "8f4ee7a64f3635bec8adadb3ed94e553751c38fc2dda033ba992ab3cb7634f8f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_dtype_like.py": {\n      "bytes": 3907,\n      "sha256": "1a299be0d4de9405780e7c09bceebd8aad17f7fbdd438071433dd0232632f4c1"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_extended_precision.py": {\n      "bytes": 434,\n      "sha256": "a649d4aa06a4d0504d33eb0444fa96fa91461fbd7f2be89e84549e7b9a108aa1"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_nbit.py": {\n      "bytes": 632,\n      "sha256": "2926cac0e2adb686fee72b53e6f095907ac3327d181efca9b534f28ffe8061cc"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_nbit_base.py": {\n      "bytes": 3058,\n      "sha256": "9cf669b1096db91e41d22680605f6a0f685efe45e79acb2ffd185a50d16c5beb"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_nested_sequence.py": {\n      "bytes": 2639,\n      "sha256": "c2678328b566bc49c1f5e61d77f982355584a1caa6068f0686929661018541ec"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_scalars.py": {\n      "bytes": 944,\n      "sha256": "2e15d8d814c799e60acde2197e98efb8c9fee5e38b8d4da7f73ef3d65e5b29ff"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_shape.py": {\n      "bytes": 275,\n      "sha256": "e9c16ff8b6d2c86f6695f4813861b2ba5f50fc652b95c1d00bd2596be9b6d1c0"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_typing/_ufunc.py": {\n      "bytes": 156,\n      "sha256": "1ce91a13eeb05747dddeb9876460b7f580072087fcb72be5cde903e32e064310"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_utils/__init__.py": {\n      "bytes": 3477,\n      "sha256": "8559d9ec2d0c09235b330f99caaf8c282612b5a197e91cea14c9e787b79de1d1"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_utils/_convertions.py": {\n      "bytes": 329,\n      "sha256": "d3133175e2cece20e61ec44cffc96e121e12fa441d328320e86c4d0c36acebd9"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_utils/_inspect.py": {\n      "bytes": 7436,\n      "sha256": "cc5b890011f4f03d4a82afde79c6240f53a0834397a75b78a2a8d9c4cd2474b9"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/_utils/_pep440.py": {\n      "bytes": 13988,\n      "sha256": "8adf4fe3fa075d6c37071768573ec93ccba3e64c45e4ceff049f19d66f67bb4c"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/char/__init__.py": {\n      "bytes": 93,\n      "sha256": "c6cea9a6b31d98d79755fb934645379c5f6a761bad59d3eee6869ea760235a67"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/conftest.py": {\n      "bytes": 8652,\n      "sha256": "3193ef75f63f7ccd1bf9c303be34026ab6a3a3c8c4bab62f3647eb485abf1c1d"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/__init__.py": {\n      "bytes": 1290,\n      "sha256": "c0935a445d5414e9d92aa881aecb2158b8d31a2119f6bbd695c8add318fb634c"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/_dtype.py": {\n      "bytes": 323,\n      "sha256": "1870617d5b6c56b3fbbf5dc8ba3133f5a18810d91821d6dfb91bbe35ec355275"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/_dtype_ctypes.py": {\n      "bytes": 351,\n      "sha256": "c17e26dfb6f4cd08319734f93a313fba3d84e42a625fd13b1cb1693ba87f9436"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/_internal.py": {\n      "bytes": 949,\n      "sha256": "ab1a472442d735471826426dd4b92d42e666e01c18bb86d79ccb8110ea7a3ce5"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/_multiarray_umath.py": {\n      "bytes": 2098,\n      "sha256": "4fcf07660143e550ae5d109278b6cfa23f7d9ca512760cb0f315b27fa7aa8714"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/_utils.py": {\n      "bytes": 923,\n      "sha256": "e5f935f09378dd183a607bda9fa42376b39e3ae2ed4652d598fe8c69d0442550"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/arrayprint.py": {\n      "bytes": 339,\n      "sha256": "2db7b8b26597605cddf6c3bd2cb2793d94b80b76d2bcbb7a1d1b70484e7ac4df"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/defchararray.py": {\n      "bytes": 347,\n      "sha256": "6bd96ebef9e2f2046b19574a3bb53fc70b0516f9339719d58312fbe632d19822"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/einsumfunc.py": {\n      "bytes": 339,\n      "sha256": "08db9c20d81422ba622d09f8c4f23f9a88308df2a5140de1ee0c00bd15706f93"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/fromnumeric.py": {\n      "bytes": 343,\n      "sha256": "e536a89c956e0b5d74aafddfddca93b668dac935f4066a89020a00267e47dd92"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/function_base.py": {\n      "bytes": 351,\n      "sha256": "be18e1cec1330ddd7544783aa6295f3093b75fa93de1a9f944026a8fe9e5ce6b"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/getlimits.py": {\n      "bytes": 335,\n      "sha256": "ea70a4e13c342e35bb8e85aca6b2392e23338ded60b0e8ceda5490fcec0107c2"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/multiarray.py": {\n      "bytes": 793,\n      "sha256": "6e374f2dbbc9ba3eb533a4d991b0793573823661f841b52aea0dde3e428fe930"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/numeric.py": {\n      "bytes": 360,\n      "sha256": "0ad93f42293207698cd31234941781f184d47d3c104977d576920c46dba76cca"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/numerictypes.py": {\n      "bytes": 347,\n      "sha256": "6d7c13c3351a8736c7ac51a14b946424ebdbe93604b109d00b9c30f6637ed55c"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/overrides.py": {\n      "bytes": 335,\n      "sha256": "d456726f453a249bb2a23b7116f43b1d2276ae97e159e73417e5f499aa428dcf"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/records.py": {\n      "bytes": 327,\n      "sha256": "f727c50f1c8e73af255ea7db6a8b2044d970d5d6d63fc0911f320f12d12d4a07"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/shape_base.py": {\n      "bytes": 339,\n      "sha256": "dacadd42d17577c2e951b0e318c5d3f93ab3d8ad8d6933be6440b8be2098b306"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/core/umath.py": {\n      "bytes": 319,\n      "sha256": "84c56636b20276a5d18a2446ed4315d06afef7162a2469a438d1909f6d222bdf"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/ctypeslib/__init__.py": {\n      "bytes": 193,\n      "sha256": "585c0c8695762c93fe2103aca5a227855f1ce973ca670aa9a44f9cbed229aaf5"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/ctypeslib/_ctypeslib.py": {\n      "bytes": 19079,\n      "sha256": "36d114a62b108437c44c12c092a61fec08ead3188d859bab6f94a61861f9e290"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/doc/ufuncs.py": {\n      "bytes": 5414,\n      "sha256": "f71b7c1f7e0686b5eb16af5c585506bc915e3daf58b87f53ab50f30279b61361"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/dtypes.py": {\n      "bytes": 1312,\n      "sha256": "cee3f0802d228c5da80d100e27a23d24a85a26e857140ca00722d06a856d4f9e"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/exceptions.py": {\n      "bytes": 7709,\n      "sha256": "df16e967d97b1779a1c2dca6be5d2fa84e346a237c6e471e6a8f1a5df66c9795"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/__init__.py": {\n      "bytes": 2448,\n      "sha256": "7008141d680941966c7eff1ca3c28136bfe6f01e9fa7374169436f25e05ffcda"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/__main__.py": {\n      "bytes": 130,\n      "sha256": "ea2da3547d9f3eb895d5aa1c4d8fdd505bd62b5f2a6bece3a6721203e3a9177c"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/__version__.py": {\n      "bytes": 48,\n      "sha256": "f7d4ba9927afba1c0698ef64bbefbb55425e921374a2de3e2747192a21dcaa9c"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/_backends/__init__.py": {\n      "bytes": 299,\n      "sha256": "eff6c0edcff10e92dce3ff2f3df1f7dbe2f19fd7dc513823436e6caddbf0bc03"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/_backends/_backend.py": {\n      "bytes": 1151,\n      "sha256": "a055d9f3e57071049b96d97cfe98162cfa82399f18fe9c7ba1e4e4fc19712537"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/_backends/_distutils.py": {\n      "bytes": 2385,\n      "sha256": "8444f4581e2acbe0f80739de906016864f78e64e707aada51940d1ea1ae25cca"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/_backends/_meson.py": {\n      "bytes": 8626,\n      "sha256": "398d3089f27dddd148e357c1aba0ddb2f9d0fdeac0e6aba2c6fa7ad1eb5a888e"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/_isocbind.py": {\n      "bytes": 2360,\n      "sha256": "cda060a5f3cd466c551b776850895b6488b207df7432c5e2c0369bad28fd1e74"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/_src_pyf.py": {\n      "bytes": 7695,\n      "sha256": "3c7a68f43dbc2aadeaff7f8a157f03dec143f5e5fc0357372ee2cdcd7cc1c8ec"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/auxfuncs.py": {\n      "bytes": 26920,\n      "sha256": "767694c2b740bf8f8b6c488735b4b5beb8d03423b4941bb25a48f746dfd48b31"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/capi_maps.py": {\n      "bytes": 30083,\n      "sha256": "fc3a1b487a90a24faa4ea2299923493c0ef37f4006316426a1b090d72ec735c7"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/cb_rules.py": {\n      "bytes": 25051,\n      "sha256": "e8a6cfbbdc9f27eee901adb8223f47df82e5d7941cf025d3a8216250923a47c6"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/cfuncs.py": {\n      "bytes": 52660,\n      "sha256": "e09e0fd76a06a725991dbd4054a025ed827741482783018c0a7075221ac99fb5"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/common_rules.py": {\n      "bytes": 5030,\n      "sha256": "ffdcb32289493068297770fde0b741b0e0e77d4b2431182ddafd37ac40881c94"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/crackfortran.py": {\n      "bytes": 146861,\n      "sha256": "642ff279b193b0600ce150718cc18577d6ff9ac5e4f167e694c9ce516034c41d"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/diagnose.py": {\n      "bytes": 5075,\n      "sha256": "616363d6f33af1ee3b2dbdbbd30959939cab714fb24e5cc669834f059ee74c05"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/f2py2e.py": {\n      "bytes": 28827,\n      "sha256": "eb0439f72fc8ab8530f5461d129265c9ba52dbf6ffc3da04b60c253c270272c3"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/f90mod_rules.py": {\n      "bytes": 9810,\n      "sha256": "ed9e6fa2b538c215f8d39c4c2faea1af88b589c09473d82be9a9f847e00c87b3"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/func2subr.py": {\n      "bytes": 10045,\n      "sha256": "0d752c605d9327127de7beb3fcc927e3bf584bda6cb56b0a52dbbdfc9e858c1f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/rules.py": {\n      "bytes": 63094,\n      "sha256": "f574d1eec4a0c245129e07119ea77ea4ddd86aad6cc230753ebacd8fb602a872"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/symbolic.py": {\n      "bytes": 53310,\n      "sha256": "a83d5d2d5d592ecb881814f794736c91abdbdfd866c2ee0311a934916928e681"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/f2py/use_rules.py": {\n      "bytes": 3376,\n      "sha256": "7a666148b3db343c811e7b1f28a5c39c6cf83ff830ae02f475f099703de7f43e"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/fft/__init__.py": {\n      "bytes": 8156,\n      "sha256": "251bd35d9a814b98e076ab2e0da2a5a7eabbfb11427a7e873ae127d5a26137c0"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/fft/_helper.py": {\n      "bytes": 6787,\n      "sha256": "8489f66721181b87cb077306bc23efa52acb5c57e1fb13b8cc62a58e4fd34236"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/fft/_pocketfft.py": {\n      "bytes": 62599,\n      "sha256": "8c4a13d3b123e678aaccacaf67c541935a7b80dd55283f2aca81bf315d02a3fa"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/fft/_pocketfft_umath.cpython-312-darwin.so": {\n      "bytes": 329912,\n      "sha256": "d902fbb947127a8e78f56d67057fad07f73563a91a60a761423e3ba09ffb8731"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/__init__.py": {\n      "bytes": 3004,\n      "sha256": "cd81aea847cfaaaecb0db89da7162cf068023409a8278c50805bc5dd7289e518"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_array_utils_impl.py": {\n      "bytes": 1697,\n      "sha256": "1985a2c8da8b43b0c6512057cf46db47a000a994ad0c8c147bbb6c6d9fffd793"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_arraypad_impl.py": {\n      "bytes": 33519,\n      "sha256": "69f17907431860505f879f7ac0f69be927464119e4068583b5f84f11fee63131"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_arraysetops_impl.py": {\n      "bytes": 37341,\n      "sha256": "9f896cc0ddc2f4f94ed7cf0b28fa5dba777ec77df3c44788afcd6448d55416b4"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_arrayterator_impl.py": {\n      "bytes": 7218,\n      "sha256": "1ed3800c81ee1bd0036db3138268783ff9ae91ed55fbc13e14d10eddb54e18f0"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_datasource.py": {\n      "bytes": 22731,\n      "sha256": "ce4f956e7e099431c455adc37ba0373608e79e2cd2262f8717466fbc0e98228e"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_format_impl.py": {\n      "bytes": 36884,\n      "sha256": "2a90cc1f9fc635b13815c166b03eda58e1fb11f8ac511e4d9b6d67d147e83691"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py": {\n      "bytes": 193738,\n      "sha256": "b2310fd26d09b566bc50c94be4d3032866503c16a75a5ff67ae994a4a7aea7c9"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_histograms_impl.py": {\n      "bytes": 38432,\n      "sha256": "52dbbb68041ced9a6c1e7ff4e288149d9ab565c7477e2a5cabd791abcd7ba155"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_index_tricks_impl.py": {\n      "bytes": 31519,\n      "sha256": "0cde783785ad86729ac3dc079a9f77ae8e1abb7d11d2cde9e35447ec91463e2b"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_iotools.py": {\n      "bytes": 30876,\n      "sha256": "d23b69be99792feff5383236d45fb58b5f6dd5edcbfbac09c517750922dec0bd"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_nanfunctions_impl.py": {\n      "bytes": 71399,\n      "sha256": "62d1c87a064ab166c6795a192ea7dd4004797f7b63d7a9a33c86cec6ca05e4b4"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_npyio_impl.py": {\n      "bytes": 98636,\n      "sha256": "557dbdca775fdf05ab660f6694c5d25194634426ddb2a22a7e1706828dc83b82"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_polynomial_impl.py": {\n      "bytes": 44125,\n      "sha256": "048794e3f17a3923deb56fdfa30544f29d03775bafb13bb57c9737b928130521"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_scimath_impl.py": {\n      "bytes": 15684,\n      "sha256": "6f9ffec91cd62483bc75afd9e01f2a41eebc8fc9693bc82db4696f18da65cadc"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_shape_base_impl.py": {\n      "bytes": 39013,\n      "sha256": "c9858af506f79e40dae1a4fee13900a06216aa78b95813d0a952156c7c37b311"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_stride_tricks_impl.py": {\n      "bytes": 19115,\n      "sha256": "d4ff05ee3167787eff06d5fbe0ebdc23a48d9f6890a733e348aa266e1355366c"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_twodim_base_impl.py": {\n      "bytes": 33923,\n      "sha256": "0fe52393e90c6565cc23ac1b8ba44fb347a02e084827309ece0468ac88b085c8"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_type_check_impl.py": {\n      "bytes": 19914,\n      "sha256": "d2b5b3bcb84c16ab9ddbeca1e7c45515063c0f61a98146039a77e99782b7b563"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_ufunclike_impl.py": {\n      "bytes": 6014,\n      "sha256": "2987a20d544c11fcfdf243a4c3e5d1cb47be8eb4f38c238399d946252afb8cd3"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_user_array_impl.py": {\n      "bytes": 8026,\n      "sha256": "65c91b14c34cdbc1136e97b5d9a6262780b58917a2a1233e4ded8dde217f5195"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_utils_impl.py": {\n      "bytes": 23499,\n      "sha256": "96db16f4c329de4d6fc3b22789d8584f97e8f60ae0c33eb6e1fbd17a21c3c967"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/_version.py": {\n      "bytes": 4783,\n      "sha256": "0affabe9bf73540e9b0febf2158f2d09746ef5de4eb069e3a976a57d8d9194e0"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/array_utils.py": {\n      "bytes": 144,\n      "sha256": "5db732849f52d0894d9cff4dcbacb280b30400259650f34e53cbdebe3d531292"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/format.py": {\n      "bytes": 477,\n      "sha256": "9e9274789853eee28d2b96b49423067e226ff91e23c8d86220f7996c970d5c1b"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/introspect.py": {\n      "bytes": 2749,\n      "sha256": "e97a1b86c7a928352ff339a83e47d53391206421afeb53f63ca97f6916c49074"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/mixins.py": {\n      "bytes": 7195,\n      "sha256": "cd047e8888c2492796c5ea4ae62fbbba34967f0163ad467fab18deac88a4c3a6"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/npyio.py": {\n      "bytes": 68,\n      "sha256": "79a3ef7c7192cd413bd132471cb3823c85fd1b98a132e0447b1cbafcf36127d4"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/recfunctions.py": {\n      "bytes": 59539,\n      "sha256": "4f869ae715dabfd9ed7f0e589b33e2abf614921d76c06938d17c812d03c21335"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/scimath.py": {\n      "bytes": 169,\n      "sha256": "aa315a41eab4cc4225ee02aa3a16a3fef9820bf29a15dc9398774b514912a792"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/stride_tricks.py": {\n      "bytes": 88,\n      "sha256": "c74fc17f097270102547706fa714e775e3fde9d1c14ff7c049b9134d3cc16202"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/lib/user_array.py": {\n      "bytes": 63,\n      "sha256": "ceceaee93017a00c9218065cd6a13a7ca0f800dfadeaead909a89968a9878aca"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/linalg/__init__.py": {\n      "bytes": 2076,\n      "sha256": "b50025b9aa09f4e77b1a67d7c03b8915a86cc1549179084da21a1a1a5d613d02"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/linalg/_linalg.py": {\n      "bytes": 114861,\n      "sha256": "e47d0e2aed21361291cd6651999a59e6479ba5866abb8eabb1dbea5b0d1618ac"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/linalg/_umath_linalg.cpython-312-darwin.so": {\n      "bytes": 154848,\n      "sha256": "86f2c3a5e1a516211e144ba13d90c18b9db903477fa66154ee5cdc4cc59c3067"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/linalg/lapack_lite.cpython-312-darwin.so": {\n      "bytes": 53008,\n      "sha256": "3f67c0ee45da7ae027d4e066495953574d9a901fe1304d9ddfbba79397a5d054"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/ma/__init__.py": {\n      "bytes": 1406,\n      "sha256": "5e90d6617c1ab83738f7e20db24e39e43d37530e29eb115db1d58e9f6aedf3b5"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/ma/core.py": {\n      "bytes": 289255,\n      "sha256": "9970fa92caa4f0c10ab51362b6347fe75d807461f98f931ef7d6daa9f30b003b"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/ma/extras.py": {\n      "bytes": 67790,\n      "sha256": "0089da9a60816d1bfae48d67d4bd416c54a6bdd42e6470e7ad9d68a2b1a84c81"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/ma/mrecords.py": {\n      "bytes": 26486,\n      "sha256": "01e5c14904591fd9946ad35ace8f23c661d3b8bf0e35ea3500c9476dab5645c4"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/ma/testutils.py": {\n      "bytes": 10213,\n      "sha256": "1d69f3dacc7244e58aedfadcd5a4768183cc36c4d4004e5e4e0b1c8d3268361c"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/matlib.py": {\n      "bytes": 10638,\n      "sha256": "fd2f4df12d8d36c1c6414725a31af1001b490de8c7894c8c74c24eed26b23e40"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/matrixlib/__init__.py": {\n      "bytes": 243,\n      "sha256": "52de88a9f8ee03e930c28e8704e6008058d75c2fe1ed857fdc7c83b0a33bd9d9"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/matrixlib/defmatrix.py": {\n      "bytes": 30875,\n      "sha256": "c29c3a95953d5faaa9f30009a240d7b76441acbd5e5ea96606dfa88c8c18ce55"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/polynomial/__init__.py": {\n      "bytes": 6726,\n      "sha256": "8064b02cda4f0a95df3e08894ac815a15b09d004b573efcc5a518e7a21b9e6c2"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/polynomial/_polybase.py": {\n      "bytes": 39358,\n      "sha256": "6f49028938149bc0f9402fcb5929bac8dbf00bbf67a4301e92c2b4bd03dc8824"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/polynomial/chebyshev.py": {\n      "bytes": 62286,\n      "sha256": "1621159a3acd55b85c6301b951012af66a910584cf86f1ab902ff2e80b314688"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/polynomial/hermite.py": {\n      "bytes": 54571,\n      "sha256": "8bc58c5e0038edc2792e214a8077556299a14326abbaceda26bb04debe096e6a"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/polynomial/hermite_e.py": {\n      "bytes": 52273,\n      "sha256": "244c341316423611ca5a98fb0b948a3f7dd04ebe88e307a04d4c8fbbd50ac8a3"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/polynomial/laguerre.py": {\n      "bytes": 52447,\n      "sha256": "2d8568238e6285355703b4af0dff07fcaac0bff2fac055dd0fbba3f830f6d7eb"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/polynomial/legendre.py": {\n      "bytes": 51102,\n      "sha256": "65667892aeceb7e239982303ed6cb910a06d23c8dcf3caac74f1b2d0c65a160f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/polynomial/polynomial.py": {\n      "bytes": 52669,\n      "sha256": "9da743f753387a41e92d56c6c5763307e83b8932bdc8ce039408e0f7926c8cff"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/polynomial/polyutils.py": {\n      "bytes": 22635,\n      "sha256": "68064957a6f465962e2b520a0cf1e188981299757f7948a48d7714a132c7c3b6"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/__init__.py": {\n      "bytes": 7480,\n      "sha256": "585ce7b73b5454d6a25c2a50967f2dc322fc1d214d4bb5c0589949b105e06ea9"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_bounded_integers.cpython-312-darwin.so": {\n      "bytes": 279912,\n      "sha256": "0e08cf005eb9fb8378086928485ce30cfb12cd2bba411b6dd7202124734c2b75"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_common.cpython-312-darwin.so": {\n      "bytes": 232000,\n      "sha256": "ecd16f24eea4c44bf28cb718300f4156cc99c1725dbeda82bb447066976cacdd"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_examples/cffi/extending.py": {\n      "bytes": 884,\n      "sha256": "8e920bd678cc85fd277a19a53079206642319ecd890bd1840d88000a15fcec6f"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_examples/cffi/parse.py": {\n      "bytes": 1750,\n      "sha256": "3caf6f754c709af76716f1f7acea609e7a484b09e277bae2e573a606b316a49c"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_examples/numba/extending.py": {\n      "bytes": 1959,\n      "sha256": "67b67f5e9ec73c4e0ae4167b030a59dbdaafb9fb45024be132d357e77b6530cc"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_examples/numba/extending_distributions.py": {\n      "bytes": 2036,\n      "sha256": "7dd78f5de523e3ac972b430ad5cb33c541d088e4e236e36cadb66a3f0e00746b"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_generator.cpython-312-darwin.so": {\n      "bytes": 651216,\n      "sha256": "391cf5d6d41959c14abe10d1108a6bfe664dd12677ae9f88abf57d35447524a8"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_mt19937.cpython-312-darwin.so": {\n      "bytes": 132176,\n      "sha256": "94083b8c1e0d99556bc732b8a59ffc273cbcb981836da95459919f3158faf475"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_pcg64.cpython-312-darwin.so": {\n      "bytes": 133456,\n      "sha256": "0230f385549bca10636d49991e15c021ced23c705bedde086fc72a6276c611b4"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_philox.cpython-312-darwin.so": {\n      "bytes": 115648,\n      "sha256": "9cebd1d7e17a5d9296bd1b6b1769466f71e773a31bf5505958cc2723af601730"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_pickle.py": {\n      "bytes": 2742,\n      "sha256": "2ede3b99afef9e72477674257398d9fc3a811ec74a8b442250d68890c7fde6a0"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/_sfc64.cpython-312-darwin.so": {\n      "bytes": 97328,\n      "sha256": "359fbd1de581fc5d26d87c9596675f9ceb35e394cc63d47812d4e6f73c5866ca"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/bit_generator.cpython-312-darwin.so": {\n      "bytes": 189920,\n      "sha256": "8a1bba22fafd2ed834c3cd7e95cc2a5a4abc09fe27004949d18e04ab2dd4d277"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/random/mtrand.cpython-312-darwin.so": {\n      "bytes": 527616,\n      "sha256": "8152e184801e01d2c92e547bece71b1eba501d6bfce6fbd3308575852e50055c"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/rec/__init__.py": {\n      "bytes": 83,\n      "sha256": "90d0186284800348b3a545516fed7cb09c3e88ab45ab6465d94e52387de91d13"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/strings/__init__.py": {\n      "bytes": 83,\n      "sha256": "a36ef01d6f2319a51f6c3a294b2126603e918dea37dc0ce4181053816ae5187e"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/testing/__init__.py": {\n      "bytes": 581,\n      "sha256": "12a7be3b1fb7252aa4e904679cf3c52c25bd224aaff4eb890e19e6dae18cdf37"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/testing/_private/__init__.py": {\n      "bytes": 0,\n      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/testing/_private/extbuild.py": {\n      "bytes": 7716,\n      "sha256": "2e0d6ca80f7843be0a8b9bbeb31d0f123eeeaebdd83f8ef4f810a2caf2701314"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/testing/_private/utils.py": {\n      "bytes": 98720,\n      "sha256": "0514e16f2db0215fed604fb3015bae396ce9ff20476c27ec395d1e7bd0786491"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/testing/overrides.py": {\n      "bytes": 2134,\n      "sha256": "07c63c3e5a6f2bbd48712ba8b9b5d68f82f935598b5529fb58c8352fbc593bc9"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/testing/print_coercion_tables.py": {\n      "bytes": 6286,\n      "sha256": "49ba0d9822dce45c95f9447c80a8c97363c8a233755d04ef1f45b30eaef93363"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/typing/__init__.py": {\n      "bytes": 6984,\n      "sha256": "658ba47953a80a0da7db6408a1a78dfda3cd538228a58dcbbf8540f66e284007"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/typing/mypy_plugin.py": {\n      "bytes": 6846,\n      "sha256": "1d7ad11a80a461471cb56e06213a01c7fa973df6721b617723a4ef9da2c54a58"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy/version.py": {\n      "bytes": 293,\n      "sha256": "833eb1051581de6bf4d23cd6f090e0fa4cbab6c548463dc718b5be62920fa2d7"\n    }\n  },\n  "numpy_metadata": {\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy-2.4.6.dist-info/METADATA": {\n      "bytes": 6608,\n      "sha256": "b082c52ccbe3b880bae177c35171b52ed6c947f3ad71e3762f2723f60a411de0"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy-2.4.6.dist-info/RECORD": {\n      "bytes": 110004,\n      "sha256": "badf350b4082ff0798860f09e89c00ecbb809948d453b1c54b5ddb951d7eee45"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/numpy-2.4.6.dist-info/WHEEL": {\n      "bytes": 122,\n      "sha256": "26726246f91715a154e07973ca8a82029674ee3cba86914a77f2e47c1a566739"\n    }\n  },\n  "numpy_total_bytes": 10961565,\n  "numpy_version": "2.4.6",\n  "original_truememory_source_config_data_read": false,\n  "original_venv_written": false,\n  "psutil_metadata": {\n    "bytes": 22187,\n    "path": "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil-7.2.2.dist-info/METADATA",\n    "sha256": "525f8ba78523cf7bc3759d62a105e7c57fd679ea702402b9449fb0746aa0217f"\n  },\n  "psutil_pins": {\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/__init__.py": {\n      "bytes": 89878,\n      "sha256": "d138a5786b163b56ba86ea0b2d5589dfca37e1bcdf8de1057fe1e933d6ab808a"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_common.py": {\n      "bytes": 25254,\n      "sha256": "1bbce9fd97e6f5439a0d385f28401f527c5a9d978d9d062593d7ef6fe215fc75"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_ntuples.py": {\n      "bytes": 10970,\n      "sha256": "9b8e8b938abae2786a515f686f4d8132218be897eb6b459b9324ed17d88a31c7"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_psaix.py": {\n      "bytes": 17616,\n      "sha256": "afe4038d5e1454559f17dac9e6c2c56ad8cf873d43880f6503e8549cc06a18ec"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_psbsd.py": {\n      "bytes": 29263,\n      "sha256": "c1a8314a9640649c99197ff261dbeec1f0080175e4a3f6b5cb370a39332deb94"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_pslinux.py": {\n      "bytes": 84787,\n      "sha256": "a8a09bab87848075b691c46112b00692745785a8c9d5a2c87037ae124b469749"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_psosx.py": {\n      "bytes": 15913,\n      "sha256": "89f0ea0e9c1fe38f224b612792bf31d07c8880d79281806e5379c00514f6b3f2"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_psposix.py": {\n      "bytes": 11575,\n      "sha256": "288887b58f5939e3b00e021f6e7b490c2f7f4852804a359918ab7d9c6fb39839"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_pssunos.py": {\n      "bytes": 23934,\n      "sha256": "cc348442008a54cb6054299b2ecae0d9a123c07bb4ba1f8e14b319487f83cee5"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_psutil_osx.abi3.so": {\n      "bytes": 105920,\n      "sha256": "ff4568b9187ca7b37c5134aaf614ffb061e7fedccebd84158343afb05991b8e2"\n    },\n    "/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_pswindows.py": {\n      "bytes": 35370,\n      "sha256": "580ad391de734c113682200ee9f4245e34737f5e26af0a73a4e8c9b4458de220"\n    }\n  },\n  "python_realpath": "/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/bin/python3.12",\n  "python_sha256": "ae711ab41ea1e6d9c2daa0ef816a0e041f9388ffd7c5bccb7d167d5b8afacfaa",\n  "python_version": "3.12.12",\n  "runtime_closure_complete": false,\n  "runtime_trust_policy": "Exact pinned package/NumPy/psutil/interpreter/framework/native dependencies; exact canonical trusted-unhashed stdlib Python and observed OS loader inputs. Qualified source proof, no native security sandbox",\n  "schema": 2,\n  "source_bytes": {\n    "truememory/__init__.py": 6932,\n    "truememory/_platform.py": 5947,\n    "truememory/client.py": 13639,\n    "truememory/clustering.py": 22412,\n    "truememory/consolidation.py": 62951,\n    "truememory/embedding_target.py": 5225,\n    "truememory/engine.py": 174355,\n    "truememory/fts_search.py": 10285,\n    "truememory/hybrid.py": 12551,\n    "truememory/hyde.py": 8143,\n    "truememory/maintenance.py": 137455,\n    "truememory/memory_budget.py": 41094,\n    "truememory/model_client.py": 34164,\n    "truememory/model_server.py": 100943,\n    "truememory/personality.py": 68823,\n    "truememory/personality_style_vec.py": 21995,\n    "truememory/predictive.py": 18593,\n    "truememory/query_classifier.py": 5045,\n    "truememory/rebuild_source.py": 19160,\n    "truememory/reranker.py": 25401,\n    "truememory/salience.py": 20238,\n    "truememory/storage.py": 68710,\n    "truememory/temporal.py": 49851,\n    "truememory/tier_config.py": 6988,\n    "truememory/tier_switch/__init__.py": 319,\n    "truememory/tier_switch/activation.py": 33404,\n    "truememory/tier_switch/cache.py": 7779,\n    "truememory/tier_switch/job.py": 25028,\n    "truememory/tier_switch/manager.py": 47952,\n    "truememory/tier_switch/runtime.py": 38531,\n    "truememory/tier_switch/sensors.py": 4336,\n    "truememory/tier_switch/serving.py": 18642,\n    "truememory/tier_switch/source.py": 28269,\n    "truememory/tier_switch/state_machine.py": 5509,\n    "truememory/tier_switch/throttler.py": 14364,\n    "truememory/tier_switch/worker.py": 27782,\n    "truememory/vector_search.py": 96703\n  },\n  "source_commit": "aeecf25f78279629789d98a29bbd8bf817233391",\n  "source_map_sha256": "e1cbacd6cdfa558110647572b614d0d3983413af3dc988530a30b56d6a06cbd6",\n  "source_root": "/Users/j/dev/truememory-maintenance-v1/.work/worktrees/whole-process-memory-297-v1",\n  "sources": {\n    "truememory/__init__.py": "00d3ab14540092b63b449d14533b17292018ae4a69cd3f3ef1345e1a02b54d98",\n    "truememory/_platform.py": "0688763a6fc26ef8db2b87866aabd5a2373a18f4ee735859cc6d595147e310c0",\n    "truememory/client.py": "f020ac930617cd5d4848775ddd7390a61665e0fbdc62d51ccaf19bccdeeb3a33",\n    "truememory/clustering.py": "ea9c2dbb7d083217983bb962f696e7bfcf08359ba5969f916fa63241baa535a5",\n    "truememory/consolidation.py": "dcff2bf4c8f80315453b5078cdae575e4414adf879b69ef8f7983e72478fd474",\n    "truememory/embedding_target.py": "da61626141f1ca5587eaa2fa3c27381f80834e79bc32351b5d7a3c20e86c7ddc",\n    "truememory/engine.py": "898f2b8dd79e8ca18154f8e4b3ef0cbfe785c27b543719f975f4a1f2cf0a010c",\n    "truememory/fts_search.py": "9c954bd2513ea8e3c82b33e0fbc311cd5b7346d12ee70da011df3f4244dc4389",\n    "truememory/hybrid.py": "17b281113b2927c6e63088c0d48dc34c2681a9943e1d4f4aa4d0a70d1c36dd17",\n    "truememory/hyde.py": "127e4d1c01767e13c0d963d6878f01a7329bb749b4510e831ef51c9c879276e4",\n    "truememory/maintenance.py": "413ffbc08f92ba3ed331cab5e0aebd67ba2cadf2e5400f8b5c09c47f80521819",\n    "truememory/memory_budget.py": "2d2578554d1c029c5205ac1fd27511f30e3d372a1178bf88cb7b81b0501409a4",\n    "truememory/model_client.py": "a8a9332ab7cf532c98594f4f7455642f00d6235d9ca8a4552a75c68a49b7d0f3",\n    "truememory/model_server.py": "424986a108d0ecd9e984ed77e352fba36305bf9aa3d79517de8fd1cfa4f8701b",\n    "truememory/personality.py": "4a55a594360435bd56152738af061988b73e3374c3e8757e934c0b88cf4cd2af",\n    "truememory/personality_style_vec.py": "f294e67b46ce4473b2cf15a49c80fc348de27f5236834dfbec832fc092c5362e",\n    "truememory/predictive.py": "88ae1a2f8436b305855a5884e3ac5ebe435106808533dc0bc563d2054d374c41",\n    "truememory/query_classifier.py": "52f1f3bfe5a75f45eb5aef389b490953422bf1d8fb03c5181bff816440bf2476",\n    "truememory/rebuild_source.py": "e46639a50d35b711bc0c0b0de9e7a90652b1386b25356c14d095551673c7bd76",\n    "truememory/reranker.py": "c02740b54246ca9f52fd6df90d424f18d3417c36734346ede128c91d3c66e90d",\n    "truememory/salience.py": "a005256ecc20c44ad7d4825a11fafa2c2b5951a762450dfce9fd80ec7e12940e",\n    "truememory/storage.py": "adc89850a897a2f9b478c8a23ad62fa438d7a214405ea21df886e029cdf4481a",\n    "truememory/temporal.py": "52e98b3924573db6a209ef27d0318e254e5c976a7a65d75762f901d712505a65",\n    "truememory/tier_config.py": "a8e690000c74f7629690c67fb99e34d4bd87f02b78795fb68ec2009f18348489",\n    "truememory/tier_switch/__init__.py": "2333416f4985986b9fcf6558478f5c3cf9f75c277961c14902162f8f11ea1705",\n    "truememory/tier_switch/activation.py": "24b820b0ff129fb42b9c13544e42b7c4b2ec8397b427d23b61c60c3e8ff19d47",\n    "truememory/tier_switch/cache.py": "77bc24c176763629f83d199cc588eca9a9e7432e36e29584a2bd3d475ac314b4",\n    "truememory/tier_switch/job.py": "c1def76a9510c3cd600ec11737ee068d71cedae2bc980048fba6c70bdee7a1f0",\n    "truememory/tier_switch/manager.py": "160a61ff2b2a4b2a7fdeb3c7fa2442ce161f5b1a426735ea3cac53829b6b8c4b",\n    "truememory/tier_switch/runtime.py": "861893308506c0189e50eea2d21a9da43b2c7be0eb90c56f8fd62e6bb40dc5be",\n    "truememory/tier_switch/sensors.py": "6471176f75ab5ddce3aa523340a6567ec24f9243c3551f4203baa8ea2f15b55b",\n    "truememory/tier_switch/serving.py": "25c1524a02affc127e170c16ceee0f61c2b3b221e28c5a19e9fb347aec57bef7",\n    "truememory/tier_switch/source.py": "f8ffe9cd5890c5e6520ca52cfe20bad49cb789d2222e29fffc4300bdbc64f227",\n    "truememory/tier_switch/state_machine.py": "39ccf8d3e944728f7bf18e6efe10d0e8f9db625f9d812220554dd980a932d1b9",\n    "truememory/tier_switch/throttler.py": "3416ad7681a22b2478e9e4375041f7d2d364b474a81255103e11abe79d5b5f32",\n    "truememory/tier_switch/worker.py": "8bae8daf54ab06c0693df46578e340cbde874dda0dccfdb5bcfc2a30d3081ef7",\n    "truememory/vector_search.py": "2acd58a0dd858cc6a3a8c69117c1666f5e74d166ed9aeb57b81da4da45804739"\n  },\n  "staged_truememory_distribution_metadata": {\n    ".work/artifacts/memory-installed-wheel-v2/metadata/METADATA": {\n      "bytes": 20005,\n      "sha256": "6f9399e0fa68584949e48662a5a9c53989eacecc78e39bb7e52cddccdd34662e"\n    },\n    ".work/artifacts/memory-installed-wheel-v2/metadata/RECORD": {\n      "bytes": 9361,\n      "sha256": "692cc8dac4cc7fb0e7a480ca1c836a5c6340fd8c1e256b789ad124ba1a02c844"\n    },\n    ".work/artifacts/memory-installed-wheel-v2/metadata/WHEEL": {\n      "bytes": 87,\n      "sha256": "5b77e4a649bbfb07fdbc123967eeecd1e5a478cfaca6eefc23c35e6fdf037848"\n    },\n    ".work/artifacts/memory-installed-wheel-v2/metadata/entry_points.txt": {\n      "bytes": 164,\n      "sha256": "0a30f6f99c5f7a3e9c87327c34fa7a56ec295a3f3237c7e084d0ca76ee668c9c"\n    },\n    ".work/artifacts/memory-installed-wheel-v2/metadata/licenses/LICENSE": {\n      "bytes": 34523,\n      "sha256": "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"\n    }\n  },\n  "staged_wheel_sha256": "f723b880e455fea1545c31a9b7fe01def1902af3a36609cfa084fc9548296c51",\n  "stdlib_complete_dependency_bytes_hashed": false,\n  "supersedes_packet_sha256": "f6418ee79b7f63783f97e9bb3cde468655b952b21279a522cc8ae5f6bfaaffe4",\n  "trusted_OS_sharedcache_bytes_hashed": false,\n  "unresolved": [\n    "Complete interpreter/stdlib/extension/dylib path admission closure requires separate review",\n    "System dylibs may be in macOS sharedcache; linkednames/OSbuild are not byte-hash proof",\n    "Actual NumPy/fullpackage64MiB/CPU2/5s fit is unproven"\n  ]\n}\n'
INTERPRETER_RAW = b'{"all_publication_held":true,"created_utc":"2026-10-12T09:02:33.473994+00:00","exact_content_files":{"/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_psutil_osx.abi3.so":{"bytes":105920,"device":16777232,"inode":13393584,"kind":"runtime native dependency","sha256":"ff4568b9187ca7b37c5134aaf614ffb061e7fedccebd84158343afb05991b8e2"},"/opt/homebrew/Cellar/mpdecimal/4.0.1/lib/libmpdec.4.0.1.dylib":{"bytes":189376,"device":16777232,"inode":5790118,"kind":"runtime native dependency","sha256":"1013bf3eafc5969ab5388fbf283b50c132b208635b59fd5b87b5c71b706aa784"},"/opt/homebrew/Cellar/openssl@3/3.6.2/lib/libcrypto.3.dylib":{"bytes":4845712,"device":16777232,"inode":5810980,"kind":"runtime native dependency","sha256":"ef2239cec921003b54b61968f00489271e2a2805d7d1b4e69d632423f38efe1e"},"/opt/homebrew/Cellar/openssl@3/3.6.2/lib/libssl.3.dylib":{"bytes":884560,"device":16777232,"inode":5810979,"kind":"runtime native dependency","sha256":"26ecfd43c742914a142c8b414b1f308cefe0531fcc6e38b6d45021649850e758"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/Python":{"bytes":6036784,"device":16777232,"inode":5801272,"kind":"CPython framework","sha256":"c365f0090d2bcd97df69c194dbe0293224ef09d88cc8c1e59260c18250e11ac6"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/bin/python3.12":{"bytes":52640,"device":16777232,"inode":5797918,"kind":"runtime native dependency","sha256":"ae711ab41ea1e6d9c2daa0ef816a0e041f9388ffd7c5bccb7d167d5b8afacfaa"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_asyncio.cpython-312-darwin.so":{"bytes":118304,"device":16777232,"inode":5798190,"kind":"runtime native dependency","sha256":"8efe7fe97e42bfa00cf4702a1a37e5e665ff9f6fa2ae45093c68a4885cffd8de"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_bisect.cpython-312-darwin.so":{"bytes":71024,"device":16777232,"inode":5798189,"kind":"runtime native dependency","sha256":"4d8df53fb16339db1e9c832d5e7559bc91fc4c7912d586051490c0cbf9506fc4"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_blake2.cpython-312-darwin.so":{"bytes":89824,"device":16777232,"inode":5798167,"kind":"runtime native dependency","sha256":"d0f8d9738e2a76eab4d1118bcc8d75af2279f841b4e21d3cf2285f85d08cee6b"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_bz2.cpython-312-darwin.so":{"bytes":73200,"device":16777232,"inode":5798204,"kind":"runtime native dependency","sha256":"1012bb5a539948ddd2732162694616155d964487d73e7e8206b25fbe0965d162"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_cn.cpython-312-darwin.so":{"bytes":186208,"device":16777232,"inode":5798191,"kind":"runtime native dependency","sha256":"097bd9bef01f21790d15dca5abf8dfa7621fffce584a53a2ab515d379b41d6ac"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_hk.cpython-312-darwin.so":{"bytes":202816,"device":16777232,"inode":5798236,"kind":"runtime native dependency","sha256":"c9a7ed2f260c8a4a3c7b3d2c2677f33bb6a5e6ed01e8ebb6b67462bf2c4bde90"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_iso2022.cpython-312-darwin.so":{"bytes":72320,"device":16777232,"inode":5798217,"kind":"runtime native dependency","sha256":"ccb4d633b05ab2ccbbde4f064a569477cbb93462eb50d07cc7642d9112097786"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_jp.cpython-312-darwin.so":{"bytes":302720,"device":16777232,"inode":5798203,"kind":"runtime native dependency","sha256":"22e89fa609becb4bfb21563299c2b5b7218a8ec6c609db86983b1d432ca529b7"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_kr.cpython-312-darwin.so":{"bytes":169776,"device":16777232,"inode":5798221,"kind":"runtime native dependency","sha256":"b75f197cc6ff439429309e3d01e7521afd88965ad3a59780c73829d54f65262d"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_tw.cpython-312-darwin.so":{"bytes":169280,"device":16777232,"inode":5798219,"kind":"runtime native dependency","sha256":"4fe57867459470fc8778fdad724992dd39a9766aa69f363cdb04021489b6383d"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_contextvars.cpython-312-darwin.so":{"bytes":68672,"device":16777232,"inode":5798232,"kind":"runtime native dependency","sha256":"3f4033c0bd82bd65ff9a0dec226069c2c15f1f61d8dcd00ee4f8200c9a2e1037"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_crypt.cpython-312-darwin.so":{"bytes":68848,"device":16777232,"inode":5798227,"kind":"runtime native dependency","sha256":"4178beda9a6f47ff4835ea7aa2c295213909ccc036785312fd9034f5180c0414"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_csv.cpython-312-darwin.so":{"bytes":91632,"device":16777232,"inode":5798179,"kind":"runtime native dependency","sha256":"369b952f7a8815f26fe0838696030a5193f5a3ce0296e7c4056ba513173776dd"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ctypes.cpython-312-darwin.so":{"bytes":158880,"device":16777232,"inode":5798205,"kind":"runtime native dependency","sha256":"4e7d8ebb1e94961ac689c37538a7e7d94661271eaea872de1956e78e003d10a5"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ctypes_test.cpython-312-darwin.so":{"bytes":74416,"device":16777232,"inode":5798187,"kind":"runtime native dependency","sha256":"279b9c93123eefa0d0b3da516123ad0d56121fb7236bf0c2cdb34f5f92bbd4c0"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_curses.cpython-312-darwin.so":{"bytes":159808,"device":16777232,"inode":5798215,"kind":"runtime native dependency","sha256":"dd35ae0d20ee960d3485ab74bc7b1475ab6bc758776528ecf3d7a11b2eb4626a"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_curses_panel.cpython-312-darwin.so":{"bytes":73312,"device":16777232,"inode":5798173,"kind":"runtime native dependency","sha256":"8948ea0a8f26f23668f0d50aa27e8a9fd08bf9a69fd9084b8d1fc5b95bdafd2f"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_datetime.cpython-312-darwin.so":{"bytes":134080,"device":16777232,"inode":5798208,"kind":"runtime native dependency","sha256":"b0cfd7a9447561997dc02726cfa4c0967c58e3e28591e019421d552fcdd9c26a"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_dbm.cpython-312-darwin.so":{"bytes":72128,"device":16777232,"inode":5798216,"kind":"runtime native dependency","sha256":"13734b3815ea839aa7da5d0df61645138aef13598f74f791385bb37d3dced490"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_decimal.cpython-312-darwin.so":{"bytes":194160,"device":16777232,"inode":5798177,"kind":"runtime native dependency","sha256":"f12c874f31c111067580a340da7aa7f528d354ab282782ea876483d4d524f263"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_elementtree.cpython-312-darwin.so":{"bytes":118352,"device":16777232,"inode":5798228,"kind":"runtime native dependency","sha256":"bc0206adb8a4cab24ed39b244fa02b8f6f6e8700ac5bf95803d42c58780367f8"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_hashlib.cpython-312-darwin.so":{"bytes":98640,"device":16777232,"inode":5798171,"kind":"runtime native dependency","sha256":"bd270aacae15d88ca2e9c63ef8011b5fd96805db6e7bb3fe281abeb3fb5650ba"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_heapq.cpython-312-darwin.so":{"bytes":69920,"device":16777232,"inode":5798222,"kind":"runtime native dependency","sha256":"2b561bbb0b50dcea90638e035574ec811531a028f7d340fed8b82c1197288eb6"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_json.cpython-312-darwin.so":{"bytes":90560,"device":16777232,"inode":5798238,"kind":"runtime native dependency","sha256":"332c8737a959c43c649d694d0cb2d5241155e01dadbb3c1cc809dbc5700d11e0"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_lsprof.cpython-312-darwin.so":{"bytes":72912,"device":16777232,"inode":5798235,"kind":"runtime native dependency","sha256":"5a6cbf0cff49c8bf7e787a99a2a4f206c47703efaf95e6143a6a3133b4a371e4"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_lzma.cpython-312-darwin.so":{"bytes":92880,"device":16777232,"inode":5798199,"kind":"runtime native dependency","sha256":"31d89874e46ad2c229c9d9a575f731fcde92f2e6cd5b0c3f2985a247f67377d4"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_md5.cpython-312-darwin.so":{"bytes":71824,"device":16777232,"inode":5798224,"kind":"runtime native dependency","sha256":"4bbf4933c4274e6de96e861662b378a7b07d3dd4ae6a078270a7d0e4b0028fa0"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_multibytecodec.cpython-312-darwin.so":{"bytes":96672,"device":16777232,"inode":5798192,"kind":"runtime native dependency","sha256":"1806b8076b5b671cad0dbbb4b7a61f0ad37e7342960b6ef71ae7c35f730cb5c7"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_multiprocessing.cpython-312-darwin.so":{"bytes":73952,"device":16777232,"inode":5798225,"kind":"runtime native dependency","sha256":"30ed202b88240c2b001166bbc7bdf009afc00faaa019d4f555d5e94d2c02d5bc"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_opcode.cpython-312-darwin.so":{"bytes":69344,"device":16777232,"inode":5798220,"kind":"runtime native dependency","sha256":"82561570fe57a1f968ef9ed3b8e54ae062037db6ca910bb5d76562c0de57d42c"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_pickle.cpython-312-darwin.so":{"bytes":152736,"device":16777232,"inode":5798202,"kind":"runtime native dependency","sha256":"9c0fef7faef33c2b0a8a9d8edb05d0f4b58bd4b54f869abed97cd84d65baf186"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_posixshmem.cpython-312-darwin.so":{"bytes":69920,"device":16777232,"inode":5798169,"kind":"runtime native dependency","sha256":"ee89ff1dbdc18da313a65d56f3d2505bd669d5491d4ef1d14c9b68c78f8915c6"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_posixsubprocess.cpython-312-darwin.so":{"bytes":72496,"device":16777232,"inode":5798195,"kind":"runtime native dependency","sha256":"5f17a49a4d6bec05a5b2127b2cf173762577175ec40236ae8fe7b242b2e222e4"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_queue.cpython-312-darwin.so":{"bytes":72576,"device":16777232,"inode":5798166,"kind":"runtime native dependency","sha256":"35aec636c3d05307a9a17f320e1bf6a38314b276d431a9ea8528578292d8d8d0"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_random.cpython-312-darwin.so":{"bytes":71888,"device":16777232,"inode":5798230,"kind":"runtime native dependency","sha256":"82a7ed4d7519482d36a3ae06953cbf117d59838d686e818b741f17421256e349"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_scproxy.cpython-312-darwin.so":{"bytes":71296,"device":16777232,"inode":5798223,"kind":"runtime native dependency","sha256":"594511ebdf53542b712a079ae41a5e9ad35d987ba43d8e6cefd4cd8ba033c3dd"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha1.cpython-312-darwin.so":{"bytes":71872,"device":16777232,"inode":5798168,"kind":"runtime native dependency","sha256":"4b6c1e2c28fc4ef9a9e16e53dcec2e5c65acb73489e32209d98f51a0e4ea35ed"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha2.cpython-312-darwin.so":{"bytes":90112,"device":16777232,"inode":5798213,"kind":"runtime native dependency","sha256":"f44daa3f719025ede5455e5dcec063d61bf3d952051c3c7ce47cd6dc05319c7a"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha3.cpython-312-darwin.so":{"bytes":73632,"device":16777232,"inode":5798181,"kind":"runtime native dependency","sha256":"0c31f8df357e9ea942c05b671e4f5449127a6c74a1f6a2f2034b678949666938"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_socket.cpython-312-darwin.so":{"bytes":132720,"device":16777232,"inode":5798176,"kind":"runtime native dependency","sha256":"39072a435f5c496bb7e0493e666b1c4fb7fd4fd3af6e737cc37973ecfb3b438b"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sqlite3.cpython-312-darwin.so":{"bytes":165664,"device":16777232,"inode":5798226,"kind":"runtime native dependency","sha256":"ed57e5b3f9a36b4fc626b9511fa85641d1bd20bddf91d51b4399d4fed12f463a"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ssl.cpython-312-darwin.so":{"bytes":220192,"device":16777232,"inode":5798183,"kind":"runtime native dependency","sha256":"b8b10518a2e538fe52f2f9c697c5c04c8f78adfbd97dd2e3e1c23698c63600fe"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_statistics.cpython-312-darwin.so":{"bytes":68864,"device":16777232,"inode":5798180,"kind":"runtime native dependency","sha256":"acc8b5429de2de1c493bfb5bf63c18a783c8aff6bc5d438dc69f3d992981fe52"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_struct.cpython-312-darwin.so":{"bytes":95008,"device":16777232,"inode":5798163,"kind":"runtime native dependency","sha256":"faad693f3025bad2bad7a846c45cbaa2e6f59311e6033ae5a014541be5ad350c"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testbuffer.cpython-312-darwin.so":{"bytes":92128,"device":16777232,"inode":5798165,"kind":"runtime native dependency","sha256":"6e5c46995d00fa2589cf3aea5321d6c82e4a5e5033c43d73a1a26adddbfd68bc"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testcapi.cpython-312-darwin.so":{"bytes":368896,"device":16777232,"inode":5798194,"kind":"runtime native dependency","sha256":"742971e338168594364e73acbddc8baed68ce803313d7e7f555ef25a374bcb42"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testclinic.cpython-312-darwin.so":{"bytes":113040,"device":16777232,"inode":5798209,"kind":"runtime native dependency","sha256":"5508c03dc49992eb4f355da7892c4d02522f6420945d1e760e6ba43447ca857b"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testimportmultiple.cpython-312-darwin.so":{"bytes":68320,"device":16777232,"inode":5798175,"kind":"runtime native dependency","sha256":"e2bc9041f0af2c72dbf97328b0cbfeb7bda6f5972bf9dbdbdfd9f66c0c9af186"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testinternalcapi.cpython-312-darwin.so":{"bytes":94576,"device":16777232,"inode":5798231,"kind":"runtime native dependency","sha256":"e94180a61ef7448e05efd93f3baff65751829ca60962dde09656b06e4adca7c0"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testmultiphase.cpython-312-darwin.so":{"bytes":77152,"device":16777232,"inode":5798211,"kind":"runtime native dependency","sha256":"7d4ae01dbd58d96ba3095b09d9714db63da59bc9327e0cc02e162d02137f52e7"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testsinglephase.cpython-312-darwin.so":{"bytes":70256,"device":16777232,"inode":5798214,"kind":"runtime native dependency","sha256":"807d2c24503b6d3ab95771ace413a2be1b2ae67bf2571b530f015ceac57abe3d"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_uuid.cpython-312-darwin.so":{"bytes":68624,"device":16777232,"inode":5798218,"kind":"runtime native dependency","sha256":"653ea323d372448beab3de0f315a364fa5b9cb324ccfe9e9427c2b525751eb7d"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxinterpchannels.cpython-312-darwin.so":{"bytes":74928,"device":16777232,"inode":5798174,"kind":"runtime native dependency","sha256":"793e428e1eac546753e8476e5a2f2d201fbbfdc58ac5688ed38d79e6fb2cf14f"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxsubinterpreters.cpython-312-darwin.so":{"bytes":73936,"device":16777232,"inode":5798188,"kind":"runtime native dependency","sha256":"0c78f075c14580405c60a57396385f0d7dc22def54729a712e58040c646090ad"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxtestfuzz.cpython-312-darwin.so":{"bytes":71328,"device":16777232,"inode":5798210,"kind":"runtime native dependency","sha256":"1bbfa1d6d2758737f7858066148406054b3303f9e72551b5873c67c99a56e168"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_zoneinfo.cpython-312-darwin.so":{"bytes":91968,"device":16777232,"inode":5798170,"kind":"runtime native dependency","sha256":"013df37ed8270df3a08c37b092440b308e14236e36f850a85cee4f00543c84b5"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/array.cpython-312-darwin.so":{"bytes":97488,"device":16777232,"inode":5798164,"kind":"runtime native dependency","sha256":"1c1eae5ebe6616222a551aab4b98e04869096570083b639a938d4108d649b916"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/audioop.cpython-312-darwin.so":{"bytes":89696,"device":16777232,"inode":5798233,"kind":"runtime native dependency","sha256":"8fb121e801de3b95ddff68203c6424094e231e0cdeb6449a0bac73b617cecfa2"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/binascii.cpython-312-darwin.so":{"bytes":89056,"device":16777232,"inode":5798237,"kind":"runtime native dependency","sha256":"c3973d4e15c7fff739fedcaabdcdcd4306ddba019dd3ed64501172bf4d0438e6"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/cmath.cpython-312-darwin.so":{"bytes":88800,"device":16777232,"inode":5798229,"kind":"runtime native dependency","sha256":"bae2872aba117e1411e7893e771a76a0b4f23c75b1db312ffb7aef23a9bf77b6"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/fcntl.cpython-312-darwin.so":{"bytes":70576,"device":16777232,"inode":5798182,"kind":"runtime native dependency","sha256":"e55c4d1ab8fde9d06febb8dd14929c80ccc536433fa426b20693a0c7c54bde5e"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/grp.cpython-312-darwin.so":{"bytes":70832,"device":16777232,"inode":5798172,"kind":"runtime native dependency","sha256":"2001fe992d994e07207f4a110e453441ec61948ddc4dff865bda6ed90a63a7e9"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/math.cpython-312-darwin.so":{"bytes":111968,"device":16777232,"inode":5798212,"kind":"runtime native dependency","sha256":"deeeae703eca73e6bf0ae53404fa4835834c4bd9bd32c889cc9c238a017a24af"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/mmap.cpython-312-darwin.so":{"bytes":73568,"device":16777232,"inode":5798207,"kind":"runtime native dependency","sha256":"f209d7eee73b2ae94faed176d23682173bf4d42a622a9eda7f57d02b1e23d262"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/nis.cpython-312-darwin.so":{"bytes":71024,"device":16777232,"inode":5798206,"kind":"runtime native dependency","sha256":"6d399051d7bbba339df38e22f2f64f789e91a917e0f69034784b7311b37fd22a"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/pyexpat.cpython-312-darwin.so":{"bytes":99600,"device":16777232,"inode":5798200,"kind":"runtime native dependency","sha256":"be320ffc44952341adfe75b8d246c24837fb5a518b4db692bedea350db3b45a0"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/readline.cpython-312-darwin.so":{"bytes":95280,"device":16777232,"inode":5798201,"kind":"runtime native dependency","sha256":"321b279033e2c37316511de74fd25a3525fbc5705793d8f50d72e36a226bd5e2"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/resource.cpython-312-darwin.so":{"bytes":70560,"device":16777232,"inode":5798234,"kind":"runtime native dependency","sha256":"5ead8d0bd513326906c217ebef38b33aa57936c19f81a8e386914d5b5d99a6a4"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/select.cpython-312-darwin.so":{"bytes":92944,"device":16777232,"inode":5798198,"kind":"runtime native dependency","sha256":"ec19dcc94817f0261ec4af51d85b509a6bcd9a65dd2676b466e3d51c55b6426e"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/syslog.cpython-312-darwin.so":{"bytes":70624,"device":16777232,"inode":5798197,"kind":"runtime native dependency","sha256":"7ccafd2aa3bd053ede95f0c632fd23f31a2904ad258290bc5330a53931fc9b8d"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/termios.cpython-312-darwin.so":{"bytes":71808,"device":16777232,"inode":5798184,"kind":"runtime native dependency","sha256":"dd7f18c041f00c51d03c9e99f31e0d73b3f49bcc04633b365bb2294c619a8f29"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/unicodedata.cpython-312-darwin.so":{"bytes":1181200,"device":16777232,"inode":5798186,"kind":"runtime native dependency","sha256":"f481050f6ae43920240605bd7af4267dc8f6f44f33f6b2328e894dd854dce399"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxlimited.cpython-312-darwin.so":{"bytes":70784,"device":16777232,"inode":5798178,"kind":"runtime native dependency","sha256":"7bad5cfecfb05b33666bdc35fdce9b23f3dff4f8658fd0f35272a5cdf3d4480c"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxlimited_35.cpython-312-darwin.so":{"bytes":70192,"device":16777232,"inode":5798193,"kind":"runtime native dependency","sha256":"4e15fa86847061606a6ab78b6e4b8aae0345f9c6d6a344cf060f673bc94869f6"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxsubtype.cpython-312-darwin.so":{"bytes":69472,"device":16777232,"inode":5798196,"kind":"runtime native dependency","sha256":"4061f5e5aa25947893b8b2281ee922e34da6826b9de409477ac472733b7ed6f8"},"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/zlib.cpython-312-darwin.so":{"bytes":94112,"device":16777232,"inode":5798185,"kind":"runtime native dependency","sha256":"94382a4afd7c5f92c30b7561cf02606ee7906427aa5739e8bda71a6fadab04de"},"/opt/homebrew/Cellar/sqlite/3.53.3/lib/libsqlite3.3.53.3.dylib":{"bytes":1286928,"device":16777232,"inode":18678560,"kind":"runtime native dependency","sha256":"b64abb706584a8952e67a3efd872e3fe6e6852c6a2a0d09adec2e94c4ab08ca5"},"/opt/homebrew/Cellar/xz/5.8.3/lib/liblzma.5.dylib":{"bytes":184512,"device":16777232,"inode":18678574,"kind":"runtime native dependency","sha256":"3d5bfa2f097c31463642b1daab5e662b44368bb4da368f85e412e7f9adcbaa10"}},"interpreter":{"framework_realpath":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/Python","realpath":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/bin/python3.12","requested":"/Users/j/dev/TrueMemory/.venv/bin/python","version":"3.12.12"},"linkage_packet_path":".work/reports/mac-native-interpreter-linkage-v1.json","linkage_packet_sha256":"762d535cdb9f021729fe101c654966e52a9d8da285d5fe77bc0b1db815d43843","live_service_queried":false,"modules_imported":false,"native_GO":false,"native_or_models_executed":false,"original_source_config_data_read":false,"runtime_closure_complete":false,"schema":1,"stdlib_complete_dependency_bytes_hashed":false,"total_pinned_native_bytes":22436608,"trust_policy":"Qualified isolated source/dependency behavior; Python audit hooks do not form a native security sandbox","trusted_OS_sharedcache_bytes_hashed":false,"trusted_stdlib":{"builtins_and_frozen":"Interpreter-provided runtime tied to executable/framework pins","canonical_root":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12","excluded_subtrees":["site-packages","test","tests","idlelib","turtledemo","ensurepip","venv"],"lib_dynload_root":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload","python_sources_byte_hashed":false},"trusted_system_loader_names":["/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate","/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation","/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit","/System/Library/Frameworks/SystemConfiguration.framework/Versions/A/SystemConfiguration","/usr/lib/libSystem.B.dylib","/usr/lib/libbz2.1.0.dylib","/usr/lib/libc++.1.dylib","/usr/lib/libedit.3.dylib","/usr/lib/libexpat.1.dylib","/usr/lib/libffi.dylib","/usr/lib/libncurses.5.4.dylib","/usr/lib/libpanel.5.4.dylib","/usr/lib/libz.1.dylib"]}\n'
LINKAGE_RAW = b'{"modules_imported":false,"observed_linkage":{"/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_psutil_osx.abi3.so":["/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation","/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/mpdecimal/4.0.1/lib/libmpdec.4.0.1.dylib":["/opt/homebrew/opt/mpdecimal/lib/libmpdec.4.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/openssl@3/3.6.2/lib/libcrypto.3.dylib":["/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/openssl@3/3.6.2/lib/libssl.3.dylib":["/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib","/opt/homebrew/Cellar/openssl@3/3.6.2/lib/libcrypto.3.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/Python":["/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/Python","/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/bin/python3.12":["/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/Python","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_asyncio.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_bisect.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_blake2.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_bz2.cpython-312-darwin.so":["/usr/lib/libbz2.1.0.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_cn.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_hk.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_iso2022.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_jp.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_kr.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_tw.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_contextvars.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_crypt.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_csv.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ctypes.cpython-312-darwin.so":["/usr/lib/libffi.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ctypes_test.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_curses.cpython-312-darwin.so":["/usr/lib/libncurses.5.4.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_curses_panel.cpython-312-darwin.so":["/usr/lib/libpanel.5.4.dylib","/usr/lib/libncurses.5.4.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_datetime.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_dbm.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_decimal.cpython-312-darwin.so":["/opt/homebrew/opt/mpdecimal/lib/libmpdec.4.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_elementtree.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_hashlib.cpython-312-darwin.so":["/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_heapq.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_json.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_lsprof.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_lzma.cpython-312-darwin.so":["/opt/homebrew/opt/xz/lib/liblzma.5.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_md5.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_multibytecodec.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_multiprocessing.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_opcode.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_pickle.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_posixshmem.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_posixsubprocess.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_queue.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_random.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_scproxy.cpython-312-darwin.so":["/System/Library/Frameworks/SystemConfiguration.framework/Versions/A/SystemConfiguration","/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha1.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha2.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha3.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_socket.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sqlite3.cpython-312-darwin.so":["/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ssl.cpython-312-darwin.so":["/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib","/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_statistics.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_struct.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testbuffer.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testcapi.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testclinic.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testimportmultiple.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testinternalcapi.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testmultiphase.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testsinglephase.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_uuid.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxinterpchannels.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxsubinterpreters.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxtestfuzz.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_zoneinfo.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/array.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/audioop.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/binascii.cpython-312-darwin.so":["/usr/lib/libz.1.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/cmath.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/fcntl.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/grp.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/math.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/mmap.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/nis.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/pyexpat.cpython-312-darwin.so":["/usr/lib/libexpat.1.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/readline.cpython-312-darwin.so":["/usr/lib/libedit.3.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/resource.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/select.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/syslog.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/termios.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/unicodedata.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxlimited.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxlimited_35.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxsubtype.cpython-312-darwin.so":["/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/zlib.cpython-312-darwin.so":["/usr/lib/libz.1.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/sqlite/3.53.3/lib/libsqlite3.3.53.3.dylib":["/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib","/usr/lib/libz.1.dylib","/usr/lib/libSystem.B.dylib"],"/opt/homebrew/Cellar/xz/5.8.3/lib/liblzma.5.dylib":["/opt/homebrew/opt/xz/lib/liblzma.5.dylib","/usr/lib/libSystem.B.dylib"]},"requested_realpath_aliases":{"/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_psutil_osx.abi3.so":"/Users/j/dev/TrueMemory/.venv/lib/python3.12/site-packages/psutil/_psutil_osx.abi3.so","/opt/homebrew/Cellar/openssl@3/3.6.2/lib/libcrypto.3.dylib":"/opt/homebrew/Cellar/openssl@3/3.6.2/lib/libcrypto.3.dylib","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/Python":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/Python","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/bin/python3.12":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/bin/python3.12","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_asyncio.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_asyncio.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_bisect.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_bisect.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_blake2.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_blake2.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_bz2.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_bz2.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_cn.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_cn.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_hk.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_hk.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_iso2022.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_iso2022.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_jp.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_jp.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_kr.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_kr.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_tw.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_codecs_tw.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_contextvars.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_contextvars.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_crypt.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_crypt.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_csv.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_csv.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ctypes.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ctypes.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ctypes_test.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ctypes_test.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_curses.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_curses.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_curses_panel.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_curses_panel.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_datetime.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_datetime.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_dbm.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_dbm.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_decimal.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_decimal.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_elementtree.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_elementtree.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_hashlib.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_hashlib.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_heapq.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_heapq.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_json.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_json.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_lsprof.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_lsprof.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_lzma.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_lzma.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_md5.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_md5.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_multibytecodec.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_multibytecodec.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_multiprocessing.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_multiprocessing.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_opcode.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_opcode.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_pickle.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_pickle.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_posixshmem.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_posixshmem.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_posixsubprocess.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_posixsubprocess.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_queue.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_queue.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_random.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_random.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_scproxy.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_scproxy.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha1.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha1.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha2.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha2.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha3.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sha3.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_socket.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_socket.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sqlite3.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_sqlite3.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ssl.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_ssl.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_statistics.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_statistics.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_struct.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_struct.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testbuffer.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testbuffer.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testcapi.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testcapi.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testclinic.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testclinic.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testimportmultiple.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testimportmultiple.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testinternalcapi.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testinternalcapi.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testmultiphase.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testmultiphase.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testsinglephase.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_testsinglephase.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_uuid.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_uuid.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxinterpchannels.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxinterpchannels.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxsubinterpreters.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxsubinterpreters.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxtestfuzz.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_xxtestfuzz.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_zoneinfo.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/_zoneinfo.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/array.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/array.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/audioop.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/audioop.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/binascii.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/binascii.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/cmath.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/cmath.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/fcntl.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/fcntl.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/grp.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/grp.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/math.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/math.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/mmap.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/mmap.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/nis.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/nis.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/pyexpat.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/pyexpat.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/readline.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/readline.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/resource.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/resource.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/select.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/select.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/syslog.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/syslog.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/termios.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/termios.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/unicodedata.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/unicodedata.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxlimited.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxlimited.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxlimited_35.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxlimited_35.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxsubtype.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/xxsubtype.cpython-312-darwin.so","/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/zlib.cpython-312-darwin.so":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/lib/python3.12/lib-dynload/zlib.cpython-312-darwin.so","/opt/homebrew/opt/mpdecimal/lib/libmpdec.4.dylib":"/opt/homebrew/Cellar/mpdecimal/4.0.1/lib/libmpdec.4.0.1.dylib","/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib":"/opt/homebrew/Cellar/openssl@3/3.6.2/lib/libcrypto.3.dylib","/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib":"/opt/homebrew/Cellar/openssl@3/3.6.2/lib/libssl.3.dylib","/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/Python":"/opt/homebrew/Cellar/python@3.12/3.12.12/Frameworks/Python.framework/Versions/3.12/Python","/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib":"/opt/homebrew/Cellar/sqlite/3.53.3/lib/libsqlite3.3.53.3.dylib","/opt/homebrew/opt/xz/lib/liblzma.5.dylib":"/opt/homebrew/Cellar/xz/5.8.3/lib/liblzma.5.dylib"},"schema":1,"sharedcache_bytes_hashed":false,"trusted_system_loader_names":["/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate","/System/Library/Frameworks/CoreFoundation.framework/Versions/A/CoreFoundation","/System/Library/Frameworks/IOKit.framework/Versions/A/IOKit","/System/Library/Frameworks/SystemConfiguration.framework/Versions/A/SystemConfiguration","/usr/lib/libSystem.B.dylib","/usr/lib/libbz2.1.0.dylib","/usr/lib/libc++.1.dylib","/usr/lib/libedit.3.dylib","/usr/lib/libexpat.1.dylib","/usr/lib/libffi.dylib","/usr/lib/libncurses.5.4.dylib","/usr/lib/libpanel.5.4.dylib","/usr/lib/libz.1.dylib"]}\n'


def policy_module() -> ModuleType:
    name = "native_package_policy_test_v4"
    spec = importlib.util.spec_from_file_location(name, PARENT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Policy source unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_module() -> ModuleType:
    name = "native_package_bootstrap_test_v4"
    spec = importlib.util.spec_from_file_location(name, BOOTSTRAP)
    if spec is None or spec.loader is None:
        raise RuntimeError("Bootstrap source unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ctypes_frames(stdlib: Path) -> tuple[ModuleType, SimpleNamespace, object]:
    filename = str(stdlib / "ctypes/__init__.py")
    module = ModuleType("ctypes")
    module.__file__ = filename
    source = "class CDLL:\n    def __init__(self):\n        pass\nclass PyDLL(CDLL):\n    pass\n"
    exec(compile(source, filename, "exec", dont_inherit=True), module.__dict__)
    instance = module.PyDLL.__new__(module.PyDLL)
    instance._name = None
    initializer = SimpleNamespace(
        f_code=SimpleNamespace(co_filename=filename, co_name="<module>"),
        f_globals=module.__dict__,
        f_back=None,
    )
    constructor = SimpleNamespace(
        f_code=module.CDLL.__init__.__code__,
        f_globals=module.__dict__,
        f_locals={"self": instance},
        f_back=initializer,
    )
    return module, constructor, instance


def manifest_fixture(policy: ModuleType) -> dict[str, object]:
    return {
        "version": 1,
        "stage": "B",
        "synthetic": True,
        "root_execution_go": True,
        "run_root": str(policy.RUN_ROOT),
        "home": str(policy.RUN_ROOT / "home"),
        "source_root": str(policy.SOURCE_ROOT),
        "source_commit": policy.SOURCE_COMMIT,
        "sources": dict(policy.SOURCE_HASHES),
        "sources_sha256": policy.SOURCE_MAP_SHA,
        "data": dict(policy.DATA_HASHES),
        "data_sha256": policy.DATA_MAP_SHA,
        "runtime_packet_sha256": policy.RUNTIME_PACKET_SHA,
        "interpreter_packet_sha256": policy.INTERPRETER_PACKET_SHA,
        "supervisor_sha256": "a" * 64,
        "bootstrap_sha256": "b" * 64,
        "limits": dict(policy.LIMITS),
        "cases": [{"id": "package-ping"}],
    }


def result_fixture(
    policy: ModuleType, pid: int = 100, uid: int = 7
) -> dict[str, object]:
    caches = {name: None for name in policy.CACHE_KEYS}
    registry = {
        "pid": pid,
        "metric": "process_rss_sampled",
        "effective_limit_bytes": 8589934592,
        "owner_count": 1,
        "closing_owner_count": 0,
        "lease_count": 0,
        "reserved_known_bytes": 0,
        "high_water_known_bytes": 512,
        "unprofiled_stages": list(policy.UNKNOWN_STAGES),
        "coverage_truncated": False,
        "physical_hard_cap": False,
        "os_enforcement": copy.deepcopy(policy.CAPABILITY),
    }
    details = {
        "package_version": "0.7.6.2",
        "initializer_executed": True,
        "paths_exact": True,
        "ping": {"ok": True, "protocol": 1},
        "timeout_ms": 200,
        "source_data_loaded": True,
        "neural_imports": [],
        "caches_before": caches,
        "caches_after": dict(caches),
        "socket_identity": {"device": 1, "inode": 2},
        "socket_uid": uid,
        "socket_mode": 384,
        "pid_file": {"pid": pid, "device": 1, "inode": 3, "mode": 384},
        "lock_identity": {"device": 1, "inode": 4},
        "bind_lock_held": True,
        "registry": registry,
    }
    return {
        "type": "result",
        "version": 1,
        "case": "package-ping",
        "passed": True,
        "proof_scope": policy.PROOF_SCOPE,
        "import_mode": "full_source_package",
        "source_commit": policy.SOURCE_COMMIT,
        "sources_sha256": policy.SOURCE_MAP_SHA,
        "source_count": 37,
        "data_sha256": policy.DATA_MAP_SHA,
        "runtime_packet_sha256": policy.RUNTIME_PACKET_SHA,
        "interpreter_packet_sha256": policy.INTERPRETER_PACKET_SHA,
        "details": details,
    }


class PackageAcceptanceContracts(unittest.TestCase):
    def listener_fixture(self) -> tuple[SimpleNamespace, Mock, Path, SimpleNamespace]:
        endpoint = self.bootstrap.RUN_ROOT / "home/.truememory/model.sock"
        listener = Mock(spec=["fileno", "getsockname", "getsockopt", "listen"])
        listener.fileno.return_value = 87
        listener.getsockname.return_value = str(endpoint)
        guard = SimpleNamespace(
            listener=listener, endpoint=endpoint, policy=self.policy
        )
        info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=7, st_nlink=1)
        return guard, listener, endpoint.with_name("model_server.pid"), info

    def test_listener_eligibility_admits_only_one_ping_candidate(self) -> None:
        guard, listener, pid_path, info = self.listener_fixture()
        with (
            patch.object(self.policy, "owned_path", return_value=info) as owned,
            patch.object(self.bootstrap.os, "getuid", return_value=7),
        ):
            self.assertIs(
                self.bootstrap.listener_eligible(guard, listener, pid_path), True
            )
            owned.assert_called_once_with(pid_path)
        listener.fileno.assert_called_once_with()
        listener.getsockname.assert_called_once_with()
        listener.getsockopt.assert_not_called()
        listener.listen.assert_not_called()

    def test_listener_eligibility_missing_or_foreign_object_refuses(self) -> None:
        guard, listener, pid_path, info = self.listener_fixture()
        for candidate in (None, Mock()):
            with (
                self.subTest(candidate=candidate),
                patch.object(self.policy, "owned_path", return_value=info) as owned,
            ):
                self.assertIs(
                    self.bootstrap.listener_eligible(guard, candidate, pid_path), False
                )
                owned.assert_not_called()
        listener.fileno.assert_not_called()
        listener.getsockname.assert_not_called()

    def test_listener_eligibility_exact_current_descriptor_refuses_invalid(
        self,
    ) -> None:
        for descriptor in (-1, 0, 1, 2, False, True, 3.0, "3", None):
            with self.subTest(descriptor=descriptor):
                guard, listener, pid_path, info = self.listener_fixture()
                listener.fileno.return_value = descriptor
                with patch.object(
                    self.policy, "owned_path", return_value=info
                ) as owned:
                    self.assertIs(
                        self.bootstrap.listener_eligible(guard, listener, pid_path),
                        False,
                    )
                    owned.assert_not_called()
                listener.getsockname.assert_not_called()

    def test_listener_eligibility_exact_endpoint_refuses_foreign_or_wrong_type(
        self,
    ) -> None:
        guard, listener, pid_path, info = self.listener_fixture()
        for endpoint in (
            str(guard.endpoint.with_name("other.sock")),
            bytes(str(guard.endpoint), "utf-8"),
            guard.endpoint,
            (str(guard.endpoint),),
            None,
        ):
            with (
                self.subTest(endpoint=endpoint),
                patch.object(self.policy, "owned_path", return_value=info) as owned,
            ):
                listener.getsockname.return_value = endpoint
                self.assertIs(
                    self.bootstrap.listener_eligible(guard, listener, pid_path), False
                )
                owned.assert_not_called()

    def test_listener_eligibility_exact_PID_path_refuses_other_paths(self) -> None:
        guard, listener, pid_path, info = self.listener_fixture()
        for other in (
            pid_path.with_name("other.pid"),
            pid_path.parent.parent / "model_server.pid",
            Path("model_server.pid"),
            pid_path.parent / "sibling/model_server.pid",
        ):
            with (
                self.subTest(path=other),
                patch.object(self.policy, "owned_path", return_value=info) as owned,
            ):
                self.assertIs(
                    self.bootstrap.listener_eligible(guard, listener, other), False
                )
                owned.assert_not_called()
        listener.fileno.assert_not_called()

    def test_listener_eligibility_private_owned_single_link_regular_evidence(
        self,
    ) -> None:
        for field, value in (
            ("st_uid", 8),
            ("st_mode", stat.S_IFREG | 0o644),
            ("st_mode", stat.S_IFDIR | 0o700),
            ("st_mode", stat.S_IFLNK | 0o600),
            ("st_nlink", 0),
            ("st_nlink", 2),
        ):
            with self.subTest(field=field, value=value):
                guard, listener, pid_path, info = self.listener_fixture()
                setattr(info, field, value)
                with (
                    patch.object(self.policy, "owned_path", return_value=info),
                    patch.object(self.bootstrap.os, "getuid", return_value=7),
                ):
                    self.assertIs(
                        self.bootstrap.listener_eligible(guard, listener, pid_path),
                        False,
                    )

    def test_listener_eligibility_actual_owned_path_canonical_and_link_checks(
        self,
    ) -> None:
        for mode in (
            "valid",
            "noncanonical",
            "link",
            "foreign_UID",
            "public_mode",
            "directory",
            "hardlink",
        ):
            with self.subTest(mode=mode):
                guard, listener, pid_path, info = self.listener_fixture()
                resolved = (
                    pid_path.with_name("other.pid")
                    if mode == "noncanonical"
                    else pid_path
                )
                if mode == "foreign_UID":
                    info.st_uid = 8
                elif mode == "public_mode":
                    info.st_mode = stat.S_IFREG | 0o644
                elif mode == "directory":
                    info.st_mode = stat.S_IFDIR | 0o700
                elif mode == "hardlink":
                    info.st_nlink = 2
                with (
                    patch.object(Path, "resolve", return_value=resolved),
                    patch.object(Path, "is_symlink", return_value=mode == "link"),
                    patch.object(Path, "lstat", return_value=info),
                    patch.object(self.bootstrap.os, "getuid", return_value=7),
                ):
                    self.assertIs(
                        self.bootstrap.listener_eligible(guard, listener, pid_path),
                        mode == "valid",
                    )

    def test_listener_eligibility_specific_source_errors_are_ineligible(self) -> None:
        for operation, error in (
            ("fileno", OSError("closed")),
            ("fileno", ValueError("closed")),
            ("getsockname", OSError(42, "Protocol not available")),
            ("owned_path", FileNotFoundError("missing")),
            ("owned_path", self.policy.SafetyError("path_not_private_owned")),
        ):
            with self.subTest(operation=operation, error=type(error)):
                guard, listener, pid_path, info = self.listener_fixture()
                if operation != "owned_path":
                    getattr(listener, operation).side_effect = error
                with patch.object(
                    self.policy,
                    "owned_path",
                    side_effect=error if operation == "owned_path" else None,
                    return_value=info,
                ):
                    self.assertIs(
                        self.bootstrap.listener_eligible(guard, listener, pid_path),
                        False,
                    )
        guard, listener, pid_path, info = self.listener_fixture()
        listener.fileno.side_effect = TypeError("Unsupported socket interface")
        with self.assertRaises(TypeError):
            self.bootstrap.listener_eligible(guard, listener, pid_path)

    def test_listener_eligibility_preserves_single_ping_and_original_bounded_loop(
        self,
    ) -> None:
        text = BOOTSTRAP.read_text()
        tree = ast.parse(text)
        worker = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "execute_case"
        )
        calls = [
            node
            for node in ast.walk(worker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_send_request"
        ]
        self.assertEqual(1, len(calls))
        self.assertEqual(
            ast.dump(
                ast.parse('model_client._send_request({"op":"ping"},timeout=0.2)')
                .body[0]
                .value
            ),
            ast.dump(calls[0]),
        )
        loop = next(node for node in worker.body if isinstance(node, ast.While))
        self.assertEqual(
            ast.dump(
                ast.parse('require(runner.is_alive(),"server_thread_stopped")').body[0]
            ),
            ast.dump(loop.body[0]),
        )
        self.assertEqual(
            ast.dump(
                ast.parse("listener_eligible(guard,listener,model_server.PID_PATH)")
                .body[0]
                .value
            ),
            ast.dump(loop.body[2].test),
        )
        self.assertEqual(
            ast.dump(
                ast.parse(
                    'require(time.monotonic()<ready_deadline,"owned_listener_deadline")'
                ).body[0]
            ),
            ast.dump(loop.body[3]),
        )
        self.assertEqual(
            ast.dump(ast.parse("time.sleep(0.001)").body[0]), ast.dump(loop.body[4])
        )
        self.assertIn("ready_deadline = time.monotonic()+0.5", text)
        self.assertNotIn("SO_ACCEPTCONN", text)

    def test_ctypes_None_equality_forgery_refuses_without_comparing(self) -> None:
        comparisons: list[object] = []

        class EqNone:
            def __eq__(self, value: object) -> bool:
                comparisons.append(value)
                return value is None

        worker = self.bootstrap
        guard = worker.AuditGuard.__new__(worker.AuditGuard)
        guard.stdlib = Path("/synthetic/stdlib")
        guard.ctypes_namespace_loads = 0
        module, frame, _ = ctypes_frames(guard.stdlib)
        with (
            patch.dict(worker.sys.modules, {"ctypes": module}),
            patch.object(worker.sys, "_getframe", return_value=frame) as frame_lookup,
        ):
            with self.assertRaises(worker.BootstrapError):
                guard("ctypes.dlopen", (EqNone(),))
            frame_lookup.assert_not_called()
        self.assertEqual([], comparisons)
        self.assertEqual(0, guard.ctypes_namespace_loads)

    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = policy_module()
        cls.bootstrap = bootstrap_module()

    def test_ctypes_actual_initializer_one_shot_only(self) -> None:
        worker = self.bootstrap
        guard = worker.AuditGuard.__new__(worker.AuditGuard)
        guard.stdlib = Path("/synthetic/stdlib")
        guard.ctypes_namespace_loads = 0
        module, frame, _ = ctypes_frames(guard.stdlib)
        with (
            patch.dict(worker.sys.modules, {"ctypes": module}),
            patch.object(worker.sys, "_getframe", return_value=frame),
        ):
            guard("ctypes.dlopen", (None,))
            self.assertEqual(1, guard.ctypes_namespace_loads)
            with self.assertRaises(worker.BootstrapError):
                guard("ctypes.dlopen", (None,))
            self.assertEqual(1, guard.ctypes_namespace_loads)

    def test_ctypes_process_namespace_argument_count_and_library_negatives(
        self,
    ) -> None:
        worker = self.bootstrap
        for args in (
            [None],
            (),
            (None, 0),
            ("/usr/lib/libSystem.B.dylib",),
            (b"library",),
            (False,),
            (0,),
        ):
            guard = worker.AuditGuard.__new__(worker.AuditGuard)
            guard.stdlib = Path("/synthetic/stdlib")
            guard.ctypes_namespace_loads = 0
            with self.subTest(args=args), self.assertRaises(worker.BootstrapError):
                guard("ctypes.dlopen", args)
            self.assertEqual(0, guard.ctypes_namespace_loads)
        for count in (False, True, 1, -1, 1.0):
            guard = worker.AuditGuard.__new__(worker.AuditGuard)
            guard.stdlib = Path("/synthetic/stdlib")
            guard.ctypes_namespace_loads = count
            with self.subTest(count=count), self.assertRaises(worker.BootstrapError):
                guard("ctypes.dlopen", (None,))

    def test_ctypes_foreign_module_globals_and_constructor_frames_refuse(self) -> None:
        worker = self.bootstrap
        for mode in (
            "module_missing",
            "foreign_module",
            "constructor_globals",
            "initializer_globals",
            "constructor_file",
            "constructor_name",
            "foreign_code",
            "initializer_file",
            "initializer_name",
            "initializer_missing",
        ):
            with self.subTest(mode=mode):
                guard = worker.AuditGuard.__new__(worker.AuditGuard)
                guard.stdlib = Path("/synthetic/stdlib")
                guard.ctypes_namespace_loads = 0
                module, frame, _ = ctypes_frames(guard.stdlib)
                current = module
                if mode == "module_missing":
                    current = None
                elif mode == "foreign_module":
                    current = ModuleType("ctypes")
                elif mode == "constructor_globals":
                    frame.f_globals = dict(module.__dict__)
                elif mode == "initializer_globals":
                    frame.f_back.f_globals = dict(module.__dict__)
                elif mode in ("constructor_file", "constructor_name", "foreign_code"):
                    source = (
                        "def __init__(self):\n    pass\n"
                        if mode != "constructor_name"
                        else "def foreign(self):\n    pass\n"
                    )
                    namespace = {}
                    filename = (
                        "/synthetic/foreign.py"
                        if mode == "constructor_file"
                        else str(guard.stdlib / "ctypes/__init__.py")
                    )
                    exec(
                        compile(source, filename, "exec", dont_inherit=True), namespace
                    )
                    frame.f_code = namespace[
                        "foreign" if mode == "constructor_name" else "__init__"
                    ].__code__
                elif mode == "initializer_file":
                    frame.f_back.f_code.co_filename = "/synthetic/foreign.py"
                elif mode == "initializer_name":
                    frame.f_back.f_code.co_name = "foreign"
                else:
                    frame.f_back = None
                with (
                    patch.dict(worker.sys.modules, {"ctypes": current}),
                    patch.object(worker.sys, "_getframe", return_value=frame),
                ):
                    with self.assertRaises(worker.BootstrapError):
                        guard("ctypes.dlopen", (None,))
                self.assertEqual(0, guard.ctypes_namespace_loads)

    def test_ctypes_namespace_type_name_and_pythonapi_negatives(self) -> None:
        worker = self.bootstrap
        for mode in (
            "CDLL_not_class",
            "CDLL_builtin_init",
            "PyDLL_not_class",
            "CDLL_self",
            "PyDLL_subclass",
            "named_library",
            "name_missing",
            "pythonapi_present",
        ):
            with self.subTest(mode=mode):
                guard = worker.AuditGuard.__new__(worker.AuditGuard)
                guard.stdlib = Path("/synthetic/stdlib")
                guard.ctypes_namespace_loads = 0
                module, frame, instance = ctypes_frames(guard.stdlib)
                if mode == "CDLL_not_class":
                    module.CDLL = None
                elif mode == "CDLL_builtin_init":
                    module.CDLL = type("CDLL", (), {})
                elif mode == "PyDLL_not_class":
                    module.PyDLL = None
                elif mode == "CDLL_self":
                    frame.f_locals["self"] = module.CDLL.__new__(module.CDLL)
                    frame.f_locals["self"]._name = None
                elif mode == "PyDLL_subclass":
                    child = type("ForeignPyDLL", (module.PyDLL,), {})
                    frame.f_locals["self"] = child.__new__(child)
                    frame.f_locals["self"]._name = None
                elif mode == "named_library":
                    instance._name = "library"
                elif mode == "name_missing":
                    del instance._name
                else:
                    module.pythonapi = None
                with (
                    patch.dict(worker.sys.modules, {"ctypes": module}),
                    patch.object(worker.sys, "_getframe", return_value=frame),
                ):
                    with self.assertRaises(worker.BootstrapError):
                        guard("ctypes.dlopen", (None,))
                self.assertEqual(0, guard.ctypes_namespace_loads)

    def test_manifest_actual_frozen_positive(self) -> None:
        value = manifest_fixture(self.policy)
        self.assertIs(self.policy.validate_manifest(value), value)
        self.assertEqual(37, len(value["sources"]))
        self.assertEqual(18, len(value))
        self.assertEqual(1, value["limits"]["threads"])
        self.assertEqual(4, self.policy.B_OS_THREAD_MAX)

    def test_manifest_bool_path_source_limit_and_extra_refuse(self) -> None:
        for field, value in (
            ("version", True),
            ("root_execution_go", 1),
            ("run_root", "/synthetic/other"),
            ("source_commit", "c" * 40),
            ("sources_sha256", "c" * 64),
            ("data_sha256", "c" * 64),
            ("runtime_packet_sha256", "c" * 64),
        ):
            with self.subTest(field=field):
                row = manifest_fixture(self.policy)
                row[field] = value
                with self.assertRaises(self.policy.SafetyError):
                    self.policy.validate_manifest(row)
        for mode in ("extra", "source_missing", "limit_bool", "limit_raise"):
            with self.subTest(mode=mode):
                row = manifest_fixture(self.policy)
                if mode == "extra":
                    row["unexpected"] = True
                elif mode == "source_missing":
                    row["sources"].pop("truememory/__init__.py")
                elif mode == "limit_bool":
                    row["limits"]["threads"] = True
                else:
                    row["limits"]["rss_abort_bytes"] += 1
                with self.assertRaises(self.policy.SafetyError):
                    self.policy.validate_manifest(row)

    def test_case_one_exact_operation(self) -> None:
        self.assertEqual(
            {"id": "package-ping"}, self.policy.case_contract({"id": "package-ping"})
        )
        for value in (
            {"id": "package-ping", "timeout": 120},
            {"id": "embed"},
            [],
            {"id": True},
        ):
            with self.assertRaises(self.policy.SafetyError):
                self.policy.case_contract(value)

    def test_runtime_actual_raw_hashes_and_qualified_flags(self) -> None:
        result = self.policy.runtime_contract(PACKAGE_RAW, INTERPRETER_RAW, LINKAGE_RAW)
        self.assertEqual(37, len(result["package"]["sources"]))
        self.assertEqual(193, len(result["package"]["numpy_files"]))
        self.assertEqual(84, len(result["interpreter"]["exact_content_files"]))
        self.assertIs(result["package"]["runtime_closure_complete"], False)
        self.assertIs(
            result["interpreter"]["stdlib_complete_dependency_bytes_hashed"], False
        )
        self.assertIs(result["linkage"]["sharedcache_bytes_hashed"], False)
        for index, raw in enumerate((PACKAGE_RAW, INTERPRETER_RAW, LINKAGE_RAW)):
            args = [PACKAGE_RAW, INTERPRETER_RAW, LINKAGE_RAW]
            args[index] = raw + b" "
            with self.assertRaises(self.policy.SafetyError):
                self.policy.runtime_contract(*args)

    def test_runtime_source_maps_data_and_metadata_current(self) -> None:
        runtime = self.policy.runtime_contract(
            PACKAGE_RAW, INTERPRETER_RAW, LINKAGE_RAW
        )
        self.assertEqual(
            "aeecf25f78279629789d98a29bbd8bf817233391",
            runtime["package"]["source_commit"],
        )
        self.assertEqual(
            {
                "truememory/data/l3_weights.json": self.policy.DATA_HASHES[
                    "truememory/data/l3_weights.json"
                ]
            },
            runtime["package"]["data"],
        )
        self.assertEqual(
            "f723b880e455fea1545c31a9b7fe01def1902af3a36609cfa084fc9548296c51",
            runtime["package"]["staged_wheel_sha256"],
        )
        self.assertTrue(
            all(
                "memory-installed-wheel-v2/metadata/" in name
                for name in runtime["package"][
                    "staged_truememory_distribution_metadata"
                ]
            )
        )
        files = self.policy.runtime_files(runtime)
        self.assertGreater(len(files), 193)

    def test_complete_result_actual_field_shape_and_intact_copy(self) -> None:
        row = result_fixture(self.policy)
        original = copy.deepcopy(row)
        self.assertIs(
            self.policy.result_contract(row, manifest_fixture(self.policy), 100, 7), row
        )
        self.assertEqual(original, row)
        self.assertEqual(
            (13, 16, 12, 7),
            (
                len(row),
                len(row["details"]),
                len(row["details"]["registry"]),
                len(row["details"]["registry"]["os_enforcement"]),
            ),
        )
        self.assertLessEqual(len(self.policy.encoded(row)) + 1, 2048)

    def test_result_bool_false_scope_cache_and_unknowns_refuse(self) -> None:
        mutations = [
            ("source_count", True),
            ("passed", 1),
            ("import_mode", "isolated_source_module"),
            ("source_commit", "a" * 40),
        ]
        for field, value in mutations:
            row = result_fixture(self.policy)
            row[field] = value
            with self.assertRaises(self.policy.SafetyError):
                self.policy.result_contract(row, manifest_fixture(self.policy), 100, 7)
        for mode in (
            "cache",
            "unknown_empty",
            "unknown_omitted",
            "cap_reason",
            "hard_cap",
            "active_lease",
            "wrong_pid",
            "protocol_bool",
        ):
            with self.subTest(mode=mode):
                row = result_fixture(self.policy)
                d = row["details"]
                r = d["registry"]
                if mode == "cache":
                    d["caches_after"]["vector_model"] = "fake"
                elif mode == "unknown_empty":
                    r["unprofiled_stages"] = []
                elif mode == "unknown_omitted":
                    r["unprofiled_stages"].pop()
                elif mode == "cap_reason":
                    r["os_enforcement"]["reason"] = ""
                elif mode == "hard_cap":
                    r["physical_hard_cap"] = True
                elif mode == "active_lease":
                    r["lease_count"] = 1
                elif mode == "wrong_pid":
                    d["pid_file"]["pid"] = 101
                else:
                    d["ping"]["protocol"] = True
                with self.assertRaises(self.policy.SafetyError):
                    self.policy.result_contract(
                        row, manifest_fixture(self.policy), 100, 7
                    )

    def test_result_large_valid_numbers_refuse_without_mutation(self) -> None:
        row = result_fixture(self.policy, pid=(1 << 63) - 1, uid=(1 << 63) - 1)
        for key in ("socket_identity", "lock_identity"):
            row["details"][key] = {"device": (1 << 63) - 1, "inode": (1 << 63) - 1}
        row["details"]["pid_file"].update(device=(1 << 63) - 1, inode=(1 << 63) - 1)
        row["details"]["registry"]["high_water_known_bytes"] = 8589934592
        original = copy.deepcopy(row)
        self.assertGreater(len(self.policy.encoded(row)) + 1, 2048)
        with self.assertRaises(self.policy.SafetyError):
            self.policy.result_contract(
                row, manifest_fixture(self.policy), (1 << 63) - 1, (1 << 63) - 1
            )
        self.assertEqual(original, row)

    def test_mutation_actual_producer_flags_and_socket_count(self) -> None:
        policy = self.policy
        control = policy.MutationPolicy({"id": "package-ping"})
        events = [
            ("os.mkdir", control.model_dir, 0o777),
            ("os.chmod", control.model_dir, 0o700),
            ("open", control.lock, os.O_CREAT | os.O_RDWR),
            ("socket.__new__", policy.ENDPOINT, 0),
            ("socket.bind", policy.ENDPOINT, 0),
            ("os.chmod", policy.ENDPOINT, 0o600),
            ("open", control.pid, os.O_CREAT | os.O_WRONLY | os.O_TRUNC),
            ("socket.__new__", policy.ENDPOINT, 0),
            ("socket.connect", policy.ENDPOINT, 0),
            ("socket.__new__", policy.ENDPOINT, 0),
        ]
        for seq, (event, path, flags) in enumerate(events, 1):
            result = control.decide(
                {
                    "type": "mutation",
                    "version": 1,
                    "seq": seq,
                    "event": event,
                    "path": str(path),
                    "source": None,
                    "flags": flags,
                }
            )
            self.assertEqual(seq, result["seq"])
        self.assertEqual(3, control.socket_objects)
        with self.assertRaises(policy.SafetyError):
            control.decide(
                {
                    "type": "mutation",
                    "version": 1,
                    "seq": 11,
                    "event": "socket.__new__",
                    "path": str(policy.ENDPOINT),
                    "source": None,
                    "flags": 0,
                }
            )

    def test_mutation_extra_flags_wrong_paths_unlink_sequence_refuse(self) -> None:
        policy = self.policy
        for event, path, flags in (
            ("os.mkdir", policy.ENDPOINT.parent, 0o700),
            ("open", policy.ENDPOINT.with_name("other"), os.O_CREAT | os.O_WRONLY),
            (
                "open",
                policy.ENDPOINT.with_name("model_server.pid"),
                os.O_CREAT | os.O_RDWR | os.O_TRUNC,
            ),
            ("os.remove", policy.ENDPOINT, 0),
            ("socket.connect", policy.ENDPOINT, 0),
        ):
            control = policy.MutationPolicy({"id": "package-ping"})
            with self.assertRaises(policy.SafetyError):
                control.decide(
                    {
                        "type": "mutation",
                        "version": 1,
                        "seq": 1,
                        "event": event,
                        "path": str(path),
                        "source": None,
                        "flags": flags,
                    }
                )
        control = policy.MutationPolicy({"id": "package-ping"})
        with self.assertRaises(policy.SafetyError):
            control.decide(
                {
                    "type": "mutation",
                    "version": 1,
                    "seq": 2,
                    "event": "os.mkdir",
                    "path": str(control.model_dir),
                    "source": None,
                    "flags": 0o777,
                }
            )

    def test_native_hash_streaming_closes_on_success_and_failure(self) -> None:
        policy = self.policy
        data = b"x" * 65537
        info = SimpleNamespace(
            st_mode=stat.S_IFREG | 0o444,
            st_size=len(data),
            st_dev=1,
            st_ino=2,
            st_mtime_ns=3,
            st_ctime_ns=4,
        )
        path = Path("/synthetic/runtime.so")
        for checksum in (hashlib.sha256(data).hexdigest(), "f" * 64):
            with (
                patch.object(Path, "resolve", lambda self: self),
                patch.object(Path, "is_symlink", return_value=False),
                patch.object(policy.os, "open", return_value=87) as opened,
                patch.object(policy.os, "fstat", return_value=info),
                patch.object(
                    policy.os, "read", side_effect=[data[:65536], data[65536:], b""]
                ) as read,
                patch.object(policy.os, "close") as close,
            ):
                if checksum.startswith("f" * 64):
                    with self.assertRaises(policy.SafetyError):
                        policy.native_file_sha256(
                            path, {"bytes": len(data), "sha256": checksum}
                        )
                else:
                    self.assertEqual(
                        checksum,
                        policy.native_file_sha256(
                            path, {"bytes": len(data), "sha256": checksum}
                        ),
                    )
                close.assert_called_once_with(87)
                self.assertTrue(opened.call_args.args[1] & os.O_NONBLOCK)
                self.assertTrue(opened.call_args.args[1] & os.O_NOFOLLOW)
                self.assertTrue(
                    all(call.args[1] <= 65536 for call in read.call_args_list)
                )

    def test_thread_samples_strict_four_ceiling(self) -> None:
        for value in (1, 2, 3, 4):
            self.assertEqual(value, self.policy.thread_sample_contract(value))
        for value in (0, 5, True, 4.0, None):
            with self.assertRaises(self.policy.SafetyError):
                self.policy.thread_sample_contract(value)
        self.assertEqual(1, self.policy.LIMITS["threads"])

    def test_terminal_ACK_whole_object(self) -> None:
        self.assertEqual(
            self.policy.TERMINAL_ACK,
            self.policy.terminal_ACK_contract(dict(self.policy.TERMINAL_ACK)),
        )
        for value in (
            {"type": "result_ack", "version": True, "case": "package-ping"},
            {"type": "result_ack", "version": 1, "case": "package-ping", "extra": 1},
            {"type": "result_ack", "version": 1, "case": "other"},
        ):
            with self.assertRaises(self.policy.SafetyError):
                self.policy.terminal_ACK_contract(value)

    def test_actual_bootstrap_accepted_descriptor_requires_exact_frame(self) -> None:
        worker = self.bootstrap
        endpoint = worker.RUN_ROOT / "home/.truememory/model.sock"
        stdlib = Path("/synthetic/stdlib")
        for mode in (
            "accepted",
            "foreign_frame",
            "foreign_listener",
            "foreign_fd",
            "regular_fd",
            "wrong_uid",
            "extra_wrapper",
        ):
            with self.subTest(mode=mode):
                guard = worker.AuditGuard.__new__(worker.AuditGuard)
                guard.stdlib, guard.endpoint = stdlib, endpoint
                guard.listener = Mock()
                guard.listener.getsockname.return_value = str(endpoint)
                guard.client = None
                guard.accepted = None
                guard.accepted_checked = False
                guard.socket_count = 3 if mode == "extra_wrapper" else 2
                guard.grant = Mock()
                caller = SimpleNamespace(
                    f_code=SimpleNamespace(
                        co_filename=str(stdlib / "socket.py"), co_name="accept"
                    ),
                    f_locals={"self": guard.listener, "fd": 87},
                    f_back=None,
                )
                if mode == "foreign_frame":
                    caller.f_code.co_filename = "/synthetic/foreign.py"
                elif mode == "foreign_listener":
                    caller.f_locals["self"] = Mock()
                elif mode == "foreign_fd":
                    caller.f_locals["fd"] = 88
                initializer = SimpleNamespace(
                    f_code=SimpleNamespace(
                        co_filename=str(stdlib / "socket.py"), co_name="__init__"
                    ),
                    f_locals={"fileno": 87},
                    f_back=caller,
                )
                info = SimpleNamespace(
                    st_mode=stat.S_IFREG if mode == "regular_fd" else stat.S_IFSOCK,
                    st_uid=8 if mode == "wrong_uid" else 7,
                    st_dev=1,
                    st_ino=2,
                )
                obj = Mock()
                with (
                    patch.object(worker.sys, "_getframe", return_value=initializer),
                    patch.object(worker.os, "fstat", return_value=info),
                    patch.object(worker.os, "getuid", return_value=7),
                ):
                    args = (
                        obj,
                        int(worker.socket.AF_UNIX),
                        int(worker.socket.SOCK_STREAM),
                        0,
                    )
                    if mode == "accepted":
                        guard.socket_new(args)
                        self.assertIs(guard.accepted, obj)
                        self.assertTrue(guard.accepted_checked)
                        self.assertEqual((1, 2), guard.accepted_identity)
                        self.assertEqual(3, guard.socket_count)
                        guard.grant.assert_called_once_with(
                            "socket.__new__", endpoint, flags=0
                        )
                    else:
                        with self.assertRaises(worker.BootstrapError):
                            guard.socket_new(args)
                        guard.grant.assert_not_called()

    def test_actual_bootstrap_grant_lock_and_whole_ACK(self) -> None:
        worker = self.bootstrap
        path = worker.RUN_ROOT / "home/.truememory/model_server.lock"
        for mode in (
            "valid",
            "lock_refused",
            "wrong_sequence",
            "extra_field",
            "terminal",
        ):
            with self.subTest(mode=mode):
                guard = worker.AuditGuard.__new__(worker.AuditGuard)
                guard.terminal = mode == "terminal"
                guard.sequence = 0
                guard.approved = {}
                guard.exchange_lock = Mock()
                guard.exchange_lock.acquire.return_value = mode != "lock_refused"
                answer = {
                    "type": "grant",
                    "version": 1,
                    "seq": 1,
                    "path": str(path),
                    "purpose": "Synthetic retained lock",
                }
                if mode == "wrong_sequence":
                    answer["seq"] = 2
                elif mode == "extra_field":
                    answer["extra"] = True
                guard.protocol = Mock()
                guard.protocol.receive.return_value = answer
                if mode == "valid":
                    guard.grant("open", path, flags=os.O_CREAT | os.O_RDWR)
                    self.assertEqual({path: os.O_RDWR}, guard.approved)
                else:
                    with self.assertRaises(worker.BootstrapError):
                        guard.grant("open", path, flags=os.O_CREAT | os.O_RDWR)
                    self.assertEqual({}, guard.approved)
                if mode == "terminal":
                    guard.exchange_lock.acquire.assert_not_called()
                    guard.protocol.emit.assert_not_called()
                else:
                    guard.exchange_lock.acquire.assert_called_once_with(timeout=0.1)
                if mode in ("terminal", "lock_refused"):
                    guard.exchange_lock.release.assert_not_called()
                    guard.protocol.emit.assert_not_called()
                else:
                    guard.exchange_lock.release.assert_called_once_with()
                    guard.protocol.receive.assert_called_once_with(2048, 0.1)
                    self.assertEqual(1, guard.protocol.emit.call_args.args[0]["seq"])

    def test_actual_bootstrap_terminal_requires_ACK_before_hard_exit(self) -> None:
        worker = self.bootstrap
        for mode in ("valid", "wrong_case", "lock_refused"):
            with self.subTest(mode=mode):
                guard = worker.AuditGuard.__new__(worker.AuditGuard)
                guard.terminal = False
                guard.policy = self.policy
                guard.exchange_lock = Mock()
                guard.exchange_lock.acquire.return_value = mode != "lock_refused"
                guard.protocol = Mock()
                guard.protocol.receive.return_value = dict(self.policy.TERMINAL_ACK)
                if mode == "wrong_case":
                    guard.protocol.receive.return_value["case"] = "other"
                with patch.object(
                    worker.os, "_exit", side_effect=RuntimeError("Inert hard exit")
                ) as hard_exit:
                    if mode == "valid":
                        with self.assertRaisesRegex(RuntimeError, "Inert hard exit"):
                            guard.terminal_result({"type": "result"})
                        hard_exit.assert_called_once_with(73)
                    elif mode == "wrong_case":
                        with self.assertRaises(self.policy.SafetyError):
                            guard.terminal_result({"type": "result"})
                        hard_exit.assert_not_called()
                    else:
                        with self.assertRaises(worker.BootstrapError):
                            guard.terminal_result({"type": "result"})
                        hard_exit.assert_not_called()
                if mode == "lock_refused":
                    guard.protocol.emit.assert_not_called()
                else:
                    guard.protocol.receive.assert_called_once_with(2048, 0.1)
                    guard.exchange_lock.release.assert_called_once_with()

    def test_original_admission_and_attempt_guards(self) -> None:
        policy = self.policy
        now = 4000000000
        rows = [
            policy.Sensor(stamp, 1, 268435456, 5)
            for stamp in (1000000000, 2000000000, 3500000000)
        ]
        policy.admission(rows, now)
        for mode in ("free", "warning", "swap", "stale"):
            changed = list(rows)
            if mode == "free":
                changed[-1] = policy.Sensor(3500000000, 1, 268435455, 5)
            elif mode == "warning":
                changed[-1] = policy.Sensor(3500000000, 2, 268435456, 5)
            elif mode == "swap":
                changed[-1] = policy.Sensor(3500000000, 1, 268435456, 6)
            else:
                changed[-1] = policy.Sensor(3000000000, 1, 268435456, 5)
            with self.assertRaises(policy.SafetyError):
                policy.admission(changed, now)
        for value in ("v1", "v2", "v99"):
            self.assertEqual(value, policy.validate_attempt_version(value))
        for value in ("v0", "v01", "v100", True):
            with self.assertRaises(policy.SafetyError):
                policy.validate_attempt_version(value)

    def test_static_worker_real_source_and_bounded_grant_terminal(self) -> None:
        tree = ast.parse(BOOTSTRAP.read_text())
        text = BOOTSTRAP.read_text()
        self.assertIn('importlib.import_module("truememory")', text)
        self.assertIn("ModelServer(automatic_rss_budget=True)", text)
        self.assertIn('_send_request({"op":"ping"},timeout=0.2)', text)
        worker = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "execute_case"
        )
        worker_body = "\n".join(
            ast.get_source_segment(text, node) for node in worker.body
        )
        self.assertNotIn("types.ModuleType", worker_body)
        self.assertIn("registry = dict(snapshot)", text)
        self.assertIn('asdict(snapshot["os_enforcement"])', text)
        self.assertIn('str(os.getpid()).encode("ascii")', text)
        self.assertIn("exchange_lock.acquire(timeout=0.1)", text)
        self.assertIn("self.policy.terminal_ACK_contract(answer)", text)
        self.assertIn("os._exit(73)", text)
        self.assertTrue(
            any(
                isinstance(node, ast.FunctionDef) and node.name == "socket_new"
                for node in ast.walk(tree)
            )
        )
        parent = PARENT.read_text()
        self.assertIn("thread_sample_contract(process.num_threads())", parent)
        self.assertIn('result.get("terminal_ACK_sent") is not True', parent)
        self.assertNotIn("((signal.SIGTERM, 0.25)", parent)


if __name__ == "__main__":
    unittest.main()
