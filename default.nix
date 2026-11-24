{
  buildPythonPackage,
  jax,
  lib,
  numpy,
  pyro-ppl,
  pytestCheckHook,
  scikit-learn,
  scipy,
  setuptools,
  torch,
  wheel,
}:
buildPythonPackage (finalAttrs: {
  pname = "sbi-simulator";
  version = "0.3.1";
  pyproject = true;

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./pyproject.toml
      ./src
      ./tests
      ./LICENSE
      ./README.md
    ];
  };

  build-system = [
    setuptools
    wheel
  ];

  dependencies = [
    numpy
    scipy
    torch
    pyro-ppl
    jax
  ];

  pythonImportsCheck = [
    "sbi_simulator"
  ];

  nativeCheckInputs = [
    pytestCheckHook
    scikit-learn
  ];

  meta = {
    description = "Collection of simulation-based inference (SBI) benchmark simulators";
    homepage = "https://github.com/Filippo-Galli/sbi-simulators";
    maintainers = with lib.maintainers; [
      Filippo-Galli
    ];
    license = lib.licenses.gpl3Plus;
  };
})
