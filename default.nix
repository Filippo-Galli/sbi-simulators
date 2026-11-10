{
  lib,
  buildPythonPackage,
  setuptools,
  wheel,
  numpy,
  scipy,
  torch,
  pyro-ppl,
  jax,
  pytestCheckHook,
}:
buildPythonPackage (finalAttrs: {
  pname = "sbi-simulator";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSource ./.;

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
  ];

  meta = {
    description = "Collection of simulation-based inference (SBI) benchmark simulators";
    homepage = "https://gitlab.inria.fr/figalli/sbi-simulator";
    maintainers = with lib.maintainers; [
      Filippo-Galli
    ];
    license = lib.licenses.gpl3Plus;
  };
})
