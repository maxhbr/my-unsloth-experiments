{
  description = "An environment for fine-tuning LLMs using unsloth on NixOS with CUDA on RTX 5090";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = let
      pkgs = import nixpkgs { 
        system = "x86_64-linux";
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };
      python = pkgs.python312;
      pythonPackages = python.pkgs;
      in pkgs.mkShell {
      buildInputs = [
        pkgs.cudaPackages.cudatoolkit
      ] ++ (with pythonPackages; [
        pytorch-bin
        unsloth
        unsloth-zoo
        torch
        # pythonPackages.torchvision
        # pythonPackages.torchaudio
        transformers
        datasets
        peft
        bitsandbytes
        accelerate
      ]);
    };
  };
}
