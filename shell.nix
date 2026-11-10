{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  packages = with pkgs; [
    nixpkgs-fmt
    python3
    ruff
    ty

    python3Packages.pytest
    python3Packages.pytest-cov
    python3Packages.setuptools
    python3Packages.wheel

    python3Packages.torch
    python3Packages.numpy
    python3Packages.scipy
    python3Packages.pyro-ppl
    python3Packages.jax
  ];
}
