{
  description = "An environment for fine-tuning LLMs using unsloth on NixOS with CUDA on RTX 5090";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = let
      pkgs = import nixpkgs { 
        system = "x86_64-linux";
        config.allowUnfree = true;
      };
      python = pkgs.python312;
      pythonPackages = python.pkgs;
      in pkgs.mkShell {
      buildInputs = [
        pkgs.cudaPackages.cudatoolkit
        pythonPackages.unsloth
        pythonPackages.unsloth-zoo
        pythonPackages.torch
        pythonPackages.torchvision
        pythonPackages.torchaudio
        pythonPackages.transformers
        pythonPackages.datasets
        pythonPackages.peft
        pythonPackages.bitsandbytes
        pythonPackages.accelerate
      ];
    };
  };
}
