#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
import subprocess
import sys

def run_command(cmd):
    """Spustí shell příkaz a vrátí jeho výstup."""
    try:
        print(f"Spouštím příkaz: {cmd}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Chyba při spuštění příkazu: {stderr.decode('utf-8')}")
            return False
        return True
    except Exception as e:
        print(f"Výjimka při spuštění příkazu: {str(e)}")
        return False

def ensure_dir(directory):
    """Zajistí, že daný adresář existuje."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def copy_model_config(source_checkpoint, target_config):
    """Načte konfiguraci modelu z checkpointu a uloží ji do cílového souboru."""
    try:
        checkpoint = torch.load(source_checkpoint, map_location=torch.device('cpu'))
        # Zde by bylo možné uložit konfiguraci modelu, ale momentálně nepotřebujeme
        return True
    except Exception as e:
        print(f"Chyba při kopírování konfigurace modelu: {str(e)}")
        return False

def run_label_gan_training(args):
    """Spustí trénink LabelGAN."""
    cmd = f"""python label_gan.py train \
        --normal_atlas_path {args.normal_atlas_path} \
        --lesion_atlas_path {args.lesion_atlas_path} \
        --label_dir {args.label_dir} \
        --output_dir {args.label_gan_output_dir} \
        --generator_filters {args.label_generator_filters} \
        --discriminator_filters {args.label_discriminator_filters} \
        --dropout_rate {args.dropout_rate} \
        --epochs {args.label_gan_epochs} \
        --batch_size {args.batch_size} \
        --lr {args.lr} \
        --beta1 {args.beta1} \
        --beta2 {args.beta2} \
        --lambda_dice {args.lambda_dice}"""
    
    if args.use_self_attention:
        cmd += " --use_self_attention"
    if args.use_spectral_norm:
        cmd += " --use_spectral_norm"
    
    return run_command(cmd)

def run_label_gan_generation(args):
    """Spustí generování LABEL map pomocí natrénovaného LabelGAN."""
    cmd = f"""python label_gan.py generate \
        --normal_atlas_path {args.normal_atlas_path} \
        --lesion_atlas_path {args.lesion_atlas_path} \
        --output_dir {args.synthetic_data_dir} \
        --checkpoint_path {args.label_gan_checkpoint} \
        --generator_filters {args.label_generator_filters} \
        --dropout_rate {args.dropout_rate} \
        --num_samples {args.num_samples}"""
    
    if args.use_self_attention:
        cmd += " --use_self_attention"
    
    return run_command(cmd)

def run_intensity_gan_training(args):
    """Spustí trénink IntensityGAN."""
    cmd = f"""python intensity_gan.py train \
        --normal_atlas_path {args.normal_atlas_path} \
        --lesion_atlas_path {args.lesion_atlas_path} \
        --zadc_dir {args.zadc_dir} \
        --label_dir {args.label_dir} \
        --output_dir {args.intensity_gan_output_dir} \
        --generator_filters {args.intensity_generator_filters} \
        --discriminator_filters {args.intensity_discriminator_filters} \
        --dropout_rate {args.dropout_rate} \
        --epochs {args.intensity_gan_epochs} \
        --batch_size {args.batch_size} \
        --lr {args.lr} \
        --beta1 {args.beta1} \
        --beta2 {args.beta2} \
        --lambda_l1 {args.lambda_l1} \
        --lambda_lesion {args.lambda_lesion} \
        --lambda_non_lesion {args.lambda_non_lesion} \
        --lambda_intensity_var {args.lambda_intensity_var}"""
    
    if args.use_self_attention:
        cmd += " --use_self_attention"
    if args.use_spectral_norm:
        cmd += " --use_spectral_norm"
    
    return run_command(cmd)

def run_intensity_gan_generation(args):
    """Spustí generování ZADC map pomocí natrénovaného IntensityGAN."""
    cmd = f"""python intensity_gan.py generate \
        --normal_atlas_path {args.normal_atlas_path} \
        --lesion_atlas_path {args.lesion_atlas_path} \
        --label_dir {os.path.join(args.synthetic_data_dir, "label")} \
        --output_dir {args.synthetic_data_dir} \
        --checkpoint_path {args.intensity_gan_checkpoint} \
        --generator_filters {args.intensity_generator_filters} \
        --dropout_rate {args.dropout_rate} \
        --num_samples {args.num_samples}"""
    
    if args.use_self_attention:
        cmd += " --use_self_attention"
    
    return run_command(cmd)

def run_synthetic_pipeline(args):
    """Spouští celý pipeline syntézy dat."""
    
    # Vytvoření potřebných adresářů
    ensure_dir(args.label_gan_output_dir)
    ensure_dir(args.intensity_gan_output_dir)
    ensure_dir(args.synthetic_data_dir)
    
    # Výběr činnosti na základě argumentů
    if args.run_label_gan_training:
        print("\n--- Spouštím trénink LabelGAN ---")
        if not run_label_gan_training(args):
            print("Chyba při tréninku LabelGAN!")
            return False
    
    if args.run_label_gan_generation:
        print("\n--- Spouštím generování LABEL map ---")
        if not run_label_gan_generation(args):
            print("Chyba při generování LABEL map!")
            return False
    
    if args.run_intensity_gan_training:
        print("\n--- Spouštím trénink IntensityGAN ---")
        if not run_intensity_gan_training(args):
            print("Chyba při tréninku IntensityGAN!")
            return False
    
    if args.run_intensity_gan_generation:
        print("\n--- Spouštím generování ZADC map ---")
        if not run_intensity_gan_generation(args):
            print("Chyba při generování ZADC map!")
            return False
    
    if args.run_complete_pipeline:
        print("\n--- Spouštím kompletní pipeline syntézy dat ---")
        
        # 1. Trénink LabelGAN
        print("\n1/4 Trénink LabelGAN...")
        if not run_label_gan_training(args):
            print("Chyba při tréninku LabelGAN!")
            return False
        
        # Automaticky nastavíme checkpoint pro generování
        label_gan_checkpoints = sorted(
            [f for f in os.listdir(args.label_gan_output_dir) if f.startswith('labelgan_checkpoint_epoch') and f.endswith('.pt')],
            key=lambda x: int(x.split('_epoch')[1].split('.pt')[0])
        )
        if label_gan_checkpoints:
            args.label_gan_checkpoint = os.path.join(args.label_gan_output_dir, label_gan_checkpoints[-1])
            print(f"Automaticky zvolen checkpoint LabelGAN: {args.label_gan_checkpoint}")
        else:
            print("Nebyl nalezen žádný checkpoint LabelGAN!")
            return False
        
        # 2. Generování LABEL map
        print("\n2/4 Generování LABEL map...")
        if not run_label_gan_generation(args):
            print("Chyba při generování LABEL map!")
            return False
        
        # 3. Trénink IntensityGAN
        print("\n3/4 Trénink IntensityGAN...")
        if not run_intensity_gan_training(args):
            print("Chyba při tréninku IntensityGAN!")
            return False
        
        # Automaticky nastavíme checkpoint pro generování
        intensity_gan_checkpoints = sorted(
            [f for f in os.listdir(args.intensity_gan_output_dir) if f.startswith('intensitygan_checkpoint_epoch') and f.endswith('.pt')],
            key=lambda x: int(x.split('_epoch')[1].split('.pt')[0])
        )
        if intensity_gan_checkpoints:
            args.intensity_gan_checkpoint = os.path.join(args.intensity_gan_output_dir, intensity_gan_checkpoints[-1])
            print(f"Automaticky zvolen checkpoint IntensityGAN: {args.intensity_gan_checkpoint}")
        else:
            print("Nebyl nalezen žádný checkpoint IntensityGAN!")
            return False
        
        # 4. Generování ZADC map
        print("\n4/4 Generování ZADC map...")
        if not run_intensity_gan_generation(args):
            print("Chyba při generování ZADC map!")
            return False
    
    print("\n--- Syntéza dat dokončena! ---")
    return True

def main():
    """Hlavní funkce skriptu"""
    parser = argparse.ArgumentParser(description="HIE Lesion Synthesis Pipeline")
    
    # Základní argumenty
    parser.add_argument('--normal_atlas_path', type=str, required=True,
                       help='Cesta k normativnímu atlasu')
    parser.add_argument('--lesion_atlas_path', type=str, default=None,
                       help='Cesta k atlasu frekvence lézí (volitelné)')
    parser.add_argument('--zadc_dir', type=str, default=None,
                       help='Adresář s registrovanými ZADC mapami pro trénink')
    parser.add_argument('--label_dir', type=str, default=None,
                       help='Adresář s registrovanými LABEL mapami pro trénink')
    
    # Výstupní adresáře
    parser.add_argument('--label_gan_output_dir', type=str, default='./output/labelgan',
                       help='Výstupní adresář pro uložení LabelGAN modelů')
    parser.add_argument('--intensity_gan_output_dir', type=str, default='./output/intensitygan',
                       help='Výstupní adresář pro uložení IntensityGAN modelů')
    parser.add_argument('--synthetic_data_dir', type=str, default='./output/synthetic_data',
                       help='Výstupní adresář pro syntetická data')
    
    # Checkpointy pro generování
    parser.add_argument('--label_gan_checkpoint', type=str, default=None,
                       help='Cesta k checkpointu LabelGAN modelu')
    parser.add_argument('--intensity_gan_checkpoint', type=str, default=None,
                       help='Cesta k checkpointu IntensityGAN modelu')
    
    # Parametry modelů
    parser.add_argument('--label_generator_filters', type=int, default=64,
                       help='Počet základních filtrů generátoru LabelGAN')
    parser.add_argument('--label_discriminator_filters', type=int, default=64,
                       help='Počet základních filtrů diskriminátoru LabelGAN')
    parser.add_argument('--intensity_generator_filters', type=int, default=64,
                       help='Počet základních filtrů generátoru IntensityGAN')
    parser.add_argument('--intensity_discriminator_filters', type=int, default=64,
                       help='Počet základních filtrů diskriminátoru IntensityGAN')
    
    # Společné parametry modelů
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                       help='Míra dropout v generátorech')
    parser.add_argument('--use_self_attention', action='store_true',
                       help='Použít self-attention mechanismus v generátorech')
    parser.add_argument('--use_spectral_norm', action='store_true',
                       help='Použít spektrální normalizaci v diskriminátorech')
    
    # Parametry tréninku
    parser.add_argument('--label_gan_epochs', type=int, default=100,
                       help='Počet epoch tréninku LabelGAN')
    parser.add_argument('--intensity_gan_epochs', type=int, default=100,
                       help='Počet epoch tréninku IntensityGAN')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Velikost dávky pro trénink')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                       help='Beta1 parametr pro Adam optimizér')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Beta2 parametr pro Adam optimizér')
    
    # Loss funkce váhy
    parser.add_argument('--lambda_dice', type=float, default=50.0,
                       help='Váha pro Dice loss (segmentace lézí)')
    parser.add_argument('--lambda_l1', type=float, default=100.0,
                       help='Váha pro L1 loss')
    parser.add_argument('--lambda_lesion', type=float, default=50.0,
                       help='Váha pro L1 loss v oblasti lézí')
    parser.add_argument('--lambda_non_lesion', type=float, default=25.0,
                       help='Váha pro L1 loss mimo oblasti lézí')
    parser.add_argument('--lambda_intensity_var', type=float, default=10.0,
                       help='Váha pro loss intenzitní variability')
    
    # Parametry generování
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Počet syntetických vzorků k vygenerování')
    
    # Přepínače pro jednotlivé kroky
    parser.add_argument('--run_label_gan_training', action='store_true',
                       help='Spustit trénink LabelGAN')
    parser.add_argument('--run_label_gan_generation', action='store_true',
                       help='Spustit generování LABEL map')
    parser.add_argument('--run_intensity_gan_training', action='store_true',
                       help='Spustit trénink IntensityGAN')
    parser.add_argument('--run_intensity_gan_generation', action='store_true',
                       help='Spustit generování ZADC map')
    parser.add_argument('--run_complete_pipeline', action='store_true',
                       help='Spustit kompletní pipeline od tréninku po generování')
    
    args = parser.parse_args()
    
    # Kontrola potřebných argumentů v závislosti na tom, co chceme dělat
    if args.run_label_gan_training or args.run_intensity_gan_training or args.run_complete_pipeline:
        if not args.label_dir:
            print("Chyba: Pro trénink je potřeba zadat --label_dir!")
            parser.print_help()
            return
        
    if args.run_intensity_gan_training or args.run_complete_pipeline:
        if not args.zadc_dir:
            print("Chyba: Pro trénink IntensityGAN je potřeba zadat --zadc_dir!")
            parser.print_help()
            return
        
    if args.run_label_gan_generation and not args.label_gan_checkpoint and not args.run_complete_pipeline:
        print("Chyba: Pro generování LABEL map je potřeba zadat --label_gan_checkpoint!")
        parser.print_help()
        return
        
    if args.run_intensity_gan_generation and not args.intensity_gan_checkpoint and not args.run_complete_pipeline:
        print("Chyba: Pro generování ZADC map je potřeba zadat --intensity_gan_checkpoint!")
        parser.print_help()
        return
    
    # Spuštění pipeline
    run_synthetic_pipeline(args)

if __name__ == "__main__":
    main() 