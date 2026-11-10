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
  ];

  shellHook = ''
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
  '';
}
