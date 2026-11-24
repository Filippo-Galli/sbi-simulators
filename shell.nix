{
  pkgs ? import <nixpkgs> { },
  sbi-simulator ? pkgs.python3Packages.callPackage ./default.nix { },
}:
pkgs.mkShell {
  inputsFrom = [ sbi-simulator ];

  packages = with pkgs; [
    nixpkgs-fmt
    python3
    ruff
    ty
    pkgs.python3Packages.corner
    pkgs.python3Packages.numpy
    pkgs.python3Packages.scikit-learn
  ];

  shellHook = ''
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
  '';
}
