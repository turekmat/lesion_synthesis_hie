import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure, morphology
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
import seaborn as sns
from tqdm import tqdm
import glob

class LesionShapeEvaluator:
    def __init__(self, real_data_dir, generated_data_dir, output_dir='./evaluation_results', num_smallest_real=20):
        """
        Inicializace evaluátoru tvarů lézí.
        
        Args:
            real_data_dir (str): Cesta k adresáři s reálnými lézemi (NIfTI soubory)
            generated_data_dir (str): Cesta k adresáři s vygenerovanými lézemi (NIfTI soubory)
            output_dir (str): Cesta pro ukládání výsledků evaluace
            num_smallest_real (int): Počet nejmenších reálných lézí, které se mají použít pro porovnání
        """
        self.real_data_dir = real_data_dir
        self.generated_data_dir = generated_data_dir
        self.output_dir = output_dir
        self.num_smallest_real = num_smallest_real
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Datové struktury pro evaluaci
        self.real_features = []
        self.real_volumes = []  # Pro ukládání objemů pro následné řazení
        self.real_file_paths = []  # Pro ukládání cest k souborům
        self.generated_features = []
        self.feature_names = [
            'volume', 'surface_area', 'sphericity', 'elongation', 
            'flatness', 'sparseness', 'fragmentation', 'complexity',
            'avg_distance_to_centroid', 'std_distance_to_centroid'
        ]
    
    def load_and_process_data(self, verbose=True):
        """Načtení a extrakce tvarových vlastností z obou datasetů, výběr nejmenších reálných lézí"""
        if verbose:
            print("Načítání a zpracování reálných lézí...")
        
        # Načtení všech reálných lézí a jejich objemů
        real_files = glob.glob(os.path.join(self.real_data_dir, "*.nii*"))
        temp_real_features = []
        temp_real_volumes = []
        temp_real_file_paths = []
        
        for file in tqdm(real_files, desc="Reálné léze"):
            try:
                lesion_data = nib.load(file).get_fdata()
                volume = np.count_nonzero(lesion_data)
                if volume > 0:  # Přeskočit prázdné léze
                    features = self.extract_shape_features(lesion_data)
                    temp_real_features.append(features)
                    temp_real_volumes.append(volume)
                    temp_real_file_paths.append(file)
            except Exception as e:
                print(f"Chyba při zpracování {file}: {e}")
        
        # Seřadit léze podle objemu (od nejmenšího po největší)
        sorted_indices = np.argsort(temp_real_volumes)
        
        # Vybrat pouze nejmenších N lézí
        selected_indices = sorted_indices[:self.num_smallest_real]
        
        # Uložit vybrané léze
        self.real_features = [temp_real_features[i] for i in selected_indices]
        self.real_file_paths = [temp_real_file_paths[i] for i in selected_indices]
        selected_volumes = [temp_real_volumes[i] for i in selected_indices]
        
        if verbose:
            print(f"Vybráno {len(self.real_features)} nejmenších reálných lézí z celkových {len(temp_real_features)}")
            print(f"Rozsah objemů vybraných lézí: {min(selected_volumes)} - {max(selected_volumes)} voxelů")
        
        if verbose:
            print("Načítání a zpracování vygenerovaných lézí...")
        
        # Načtení vygenerovaných lézí
        generated_files = glob.glob(os.path.join(self.generated_data_dir, "*.nii*"))
        for file in tqdm(generated_files, desc="Vygenerované léze"):
            try:
                lesion_data = nib.load(file).get_fdata()
                if np.count_nonzero(lesion_data) > 0:  # Přeskočit prázdné léze
                    features = self.extract_shape_features(lesion_data)
                    self.generated_features.append(features)
            except Exception as e:
                print(f"Chyba při zpracování {file}: {e}")
        
        # Převod na numpy pole pro snazší práci
        self.real_features = np.array(self.real_features)
        self.generated_features = np.array(self.generated_features)
        
        if verbose:
            print(f"Zpracováno {len(self.real_features)} vybraných reálných a {len(self.generated_features)} vygenerovaných lézí")
    
    def extract_shape_features(self, lesion_data):
        """
        Extrahuje tvarové vlastnosti z binární léze.
        
        Args:
            lesion_data (np.ndarray): 3D binární maska léze
        
        Returns:
            list: Vektor tvarových vlastností
        """
        # Převod na binární masku
        binary_lesion = lesion_data > 0
        
        # Základní vlastnosti
        volume = np.count_nonzero(binary_lesion)
        
        # Výpočet labelů pro propojené komponenty
        labeled, num_components = ndimage.label(binary_lesion)
        
        # Získat největší komponentu, pokud je jich více
        if num_components > 1:
            sizes = np.bincount(labeled.ravel())[1:]
            largest_label = sizes.argmax() + 1
            binary_lesion = (labeled == largest_label)
        
        # Extrakce 3D vlastností
        props = measure.regionprops(binary_lesion.astype(int))[0]
        
        # Výpočet objemu, povrchu a tvaru
        surface_area = self._compute_surface_area(binary_lesion)
        
        # Sféricita: 1 pro dokonalou kouli, menší pro komplexnější tvary
        sphericity = (36 * np.pi * volume**2)**(1/3) / surface_area if surface_area > 0 else 0
        
        # Elongace a zploštělost z hlavních os
        if hasattr(props, 'principal_inertia_components'):
            # V novějších verzích skimage
            eigen_values = props.principal_inertia_components
        else:
            # V starších verzích skimage
            eigen_values = props.inertia_tensor_eigvals
        
        if len(eigen_values) >= 3 and eigen_values[2] > 0:
            elongation = eigen_values[0] / eigen_values[2]
            flatness = eigen_values[1] / eigen_values[2]
        else:
            elongation = 1.0
            flatness = 1.0
        
        # Řídkost (sparseness) - měří, jak moc je léze "děravá"
        convex_hull = morphology.convex_hull_image(binary_lesion)
        convex_volume = np.count_nonzero(convex_hull)
        sparseness = volume / convex_volume if convex_volume > 0 else 1.0
        
        # Fragmentace - počet komponent relativně k objemu
        fragmentation = num_components / np.cbrt(volume) if volume > 0 else 0
        
        # Komplexita - složitost povrchu vzhledem k objemu
        complexity = surface_area / (np.cbrt(volume)**2) if volume > 0 else 0
        
        # Vzdálenosti k centroidu
        if volume > 0:
            # Najít souřadnice voxelů léze
            coords = np.array(np.where(binary_lesion)).T
            centroid = np.mean(coords, axis=0)
            
            # Vypočítat vzdálenosti od centroidu ke každému voxelu
            distances = np.sqrt(np.sum((coords - centroid)**2, axis=1))
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
        else:
            avg_distance = 0
            std_distance = 0
        
        # Vrátit vektor vlastností
        return [
            volume, surface_area, sphericity, elongation, 
            flatness, sparseness, fragmentation, complexity,
            avg_distance, std_distance
        ]
    
    def _compute_surface_area(self, binary_volume):
        """Výpočet povrchové plochy 3D binárního objektu pomocí gradientní metody"""
        # Použití gradientu pro nalezení hran
        grad_x = ndimage.sobel(binary_volume.astype(float), axis=0)
        grad_y = ndimage.sobel(binary_volume.astype(float), axis=1)
        grad_z = ndimage.sobel(binary_volume.astype(float), axis=2)
        
        # Výpočet magnitudy gradientu
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Povrchové voxely jsou ty, kde je nenulová hodnota magnitudy gradientu
        return np.count_nonzero(gradient_magnitude)
    
    def compute_shape_probability(self, normalize=True):
        """
        Vypočítá pravděpodobnost tvarů vygenerovaných lézí vzhledem k distribuci reálných lézí.
        
        Args:
            normalize (bool): Zda normalizovat vlastnosti před výpočtem
            
        Returns:
            np.ndarray: Vektor pravděpodobností pro každou generovanou lézi
        """
        if len(self.real_features) == 0 or len(self.generated_features) == 0:
            raise ValueError("Nejprve je třeba načíst data pomocí load_and_process_data()")
        
        # Škálování vlastností pro stabilnější výsledky
        if normalize:
            scaler = StandardScaler()
            real_features_scaled = scaler.fit_transform(self.real_features)
            gen_features_scaled = scaler.transform(self.generated_features)
        else:
            real_features_scaled = self.real_features
            gen_features_scaled = self.generated_features
        
        # Metoda 1: Isolation Forest pro detekci anomálií (outlierů)
        clf = IsolationForest(contamination=0.1, random_state=42)
        clf.fit(real_features_scaled)
        
        # Vyšší skóre = podobnější reálným datům
        anomaly_scores = clf.score_samples(gen_features_scaled)
        
        # Metoda 2: Kernel Density Estimation pro odhad pravděpodobnosti
        kde = KernelDensity(bandwidth=0.5, kernel='gaussian')
        kde.fit(real_features_scaled)
        
        # Log pravděpodobnosti
        log_probs = kde.score_samples(gen_features_scaled)
        
        # Převod na pravděpodobnosti (exp z log pravděpodobností)
        probs = np.exp(log_probs)
        
        # Kombinace obou metod pro robustnější hodnocení
        # Normalizace obou metrik na rozsah [0, 1]
        anomaly_scores_norm = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores) + 1e-10)
        probs_norm = (probs - np.min(probs)) / (np.max(probs) - np.min(probs) + 1e-10)
        
        # Průměr obou normalizovaných metrik
        combined_probs = (anomaly_scores_norm + probs_norm) / 2.0
        
        return combined_probs
    
    def evaluate_distribution_similarity(self):
        """
        Vyhodnocení podobnosti distribucí tvarových vlastností mezi reálnými a vygenerovanými lézemi.
        
        Returns:
            dict: Slovník s metrikami podobnosti pro každou vlastnost
        """
        if len(self.real_features) == 0 or len(self.generated_features) == 0:
            raise ValueError("Nejprve je třeba načíst data pomocí load_and_process_data()")
        
        results = {}
        
        for i, feature_name in enumerate(self.feature_names):
            real_feature = self.real_features[:, i]
            gen_feature = self.generated_features[:, i]
            
            # Kolmogorov-Smirnov test - měří maximální rozdíl mezi kumulativními distribucemi
            ks_stat, ks_pval = ks_2samp(real_feature, gen_feature)
            
            # Wasserstein distance (Earth Mover's Distance) - měří "cenu" transformace jedné distribuce na druhou
            w_distance = wasserstein_distance(real_feature, gen_feature)
            
            results[feature_name] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'wasserstein_distance': w_distance,
                # p-hodnota < 0.05 znamená, že distribuce jsou statisticky odlišné
                'is_similar': ks_pval >= 0.05
            }
        
        return results
    
    def visualize_results(self):
        """Vytvoří vizualizace pro porovnání tvarových vlastností a pravděpodobností"""
        if len(self.real_features) == 0 or len(self.generated_features) == 0:
            raise ValueError("Nejprve je třeba načíst data pomocí load_and_process_data()")
        
        # 1. Distribuce tvarových vlastností
        fig, axes = plt.subplots(5, 2, figsize=(15, 20))
        axes = axes.flatten()
        
        for i, feature_name in enumerate(self.feature_names):
            ax = axes[i]
            
            # Použití KDE pro hladší vizualizaci distribucí
            sns.kdeplot(self.real_features[:, i], ax=ax, label='Reálné léze', color='blue')
            sns.kdeplot(self.generated_features[:, i], ax=ax, label='Vygenerované léze', color='red')
            
            ax.set_title(f'Distribuce: {feature_name}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'shape_feature_distributions.png'), dpi=300)
        
        # 2. Scatter plot pro nejdůležitější vlastnosti (volume vs. sphericity, complexity vs. fragmentation)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Volume vs. Sphericity
        ax1.scatter(self.real_features[:, 0], self.real_features[:, 2], 
                   alpha=0.5, label='Reálné léze', color='blue')
        ax1.scatter(self.generated_features[:, 0], self.generated_features[:, 2], 
                   alpha=0.5, label='Vygenerované léze', color='red')
        ax1.set_xlabel('Objem')
        ax1.set_ylabel('Sféricita')
        ax1.legend()
        ax1.set_title('Objem vs. Sféricita')
        
        # Complexity vs. Fragmentation
        ax2.scatter(self.real_features[:, 7], self.real_features[:, 6], 
                   alpha=0.5, label='Reálné léze', color='blue')
        ax2.scatter(self.generated_features[:, 7], self.generated_features[:, 6], 
                   alpha=0.5, label='Vygenerované léze', color='red')
        ax2.set_xlabel('Komplexita')
        ax2.set_ylabel('Fragmentace')
        ax2.legend()
        ax2.set_title('Komplexita vs. Fragmentace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'shape_feature_scatterplots.png'), dpi=300)
        
        # 3. Distribuce pravděpodobností tvarů
        shape_probs = self.compute_shape_probability()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(shape_probs, bins=30, kde=True)
        plt.axvline(0.5, color='r', linestyle='--', label='Hranice pravděpodobnosti (0.5)')
        plt.title('Distribuce pravděpodobností tvarů vygenerovaných lézí')
        plt.xlabel('Pravděpodobnost (vyšší = realističtější tvar)')
        plt.ylabel('Četnost')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'shape_probability_distribution.png'), dpi=300)
        
        # 4. Heatmapa podobnosti distribucí
        similarity_results = self.evaluate_distribution_similarity()
        
        # Extrakce hodnot pro heatmapu
        feature_names = list(similarity_results.keys())
        ks_pvalues = [similarity_results[f]['ks_pvalue'] for f in feature_names]
        w_distances = [similarity_results[f]['wasserstein_distance'] for f in feature_names]
        
        # Normalizace Wasserstein distances pro lepší vizualizaci
        w_distances_norm = w_distances / np.max(w_distances)
        
        # Vytvoření DataFrame pro heatmapu
        heatmap_data = pd.DataFrame({
            'Feature': feature_names,
            'KS p-value': ks_pvalues,
            'Normalized Wasserstein Distance': w_distances_norm
        }).set_index('Feature')
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='coolwarm_r', linewidths=0.5)
        plt.title('Podobnost distribucí tvarových vlastností')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'distribution_similarity_heatmap.png'), dpi=300)
    
    def print_evaluation_results(self):
        """Tiskne souhrnný report o všech evaluacích místo generování HTML"""
        if len(self.real_features) == 0 or len(self.generated_features) == 0:
            raise ValueError("Nejprve je třeba načíst data pomocí load_and_process_data()")
        
        # Výpočet pravděpodobností tvarů
        shape_probs = self.compute_shape_probability()
        
        # Vyhodnocení podobnosti distribucí
        dist_similarity = self.evaluate_distribution_similarity()
        
        # Základní statistiky pravděpodobností
        prob_mean = np.mean(shape_probs)
        prob_median = np.median(shape_probs)
        prob_std = np.std(shape_probs)
        prob_min = np.min(shape_probs)
        prob_max = np.max(shape_probs)
        
        # Počet lézí s vysokou pravděpodobností (> 0.7)
        high_prob_count = np.sum(shape_probs > 0.7)
        high_prob_percentage = (high_prob_count / len(shape_probs)) * 100
        
        # Počet lézí s nízkou pravděpodobností (< 0.3)
        low_prob_count = np.sum(shape_probs < 0.3)
        low_prob_percentage = (low_prob_count / len(shape_probs)) * 100
        
        # Tisk výsledků
        print("\n" + "="*80)
        print(f" EVALUAČNÍ REPORT TVARŮ LÉZÍ - {self.num_smallest_real} NEJMENŠÍCH REÁLNÝCH LÉZÍ")
        print("="*80)
        
        print("\nREÁLNÉ LÉZE POUŽITÉ PRO POROVNÁNÍ:")
        for i, file_path in enumerate(self.real_file_paths):
            print(f"  {i+1}. {os.path.basename(file_path)} - Objem: {self.real_features[i, 0]:.0f} voxelů")
        
        print("\nSTATISTIKY PRAVDĚPODOBNOSTI TVARŮ:")
        print(f"  Průměrná pravděpodobnost: {prob_mean:.4f}")
        print(f"  Mediánová pravděpodobnost: {prob_median:.4f}")
        print(f"  Směrodatná odchylka: {prob_std:.4f}")
        print(f"  Minimální pravděpodobnost: {prob_min:.4f}")
        print(f"  Maximální pravděpodobnost: {prob_max:.4f}")
        print(f"  Léze s vysokou pravděpodobností (>0.7): {high_prob_count} ({high_prob_percentage:.1f}%)")
        print(f"  Léze s nízkou pravděpodobností (<0.3): {low_prob_count} ({low_prob_percentage:.1f}%)")
        
        # Výstup podobnosti distribucí
        print("\nPODOBNOST DISTRIBUCÍ TVAROVÝCH VLASTNOSTÍ:")
        print("  Kolmogorov-Smirnov p-hodnota > 0.05 znamená, že distribuce jsou statisticky podobné.\n")
        
        # Tvorba tabulky pro konzoli
        header = f"{'Vlastnost':<20} | {'KS p-hodnota':<15} | {'Wasserstein vzd.':<15} | {'Je podobná?':<12}"
        print("  " + header)
        print("  " + "-"*len(header))
        
        for feature, stats in dist_similarity.items():
            is_similar = stats['is_similar']
            similarity_text = "Ano" if is_similar else "Ne"
            print(f"  {feature:<20} | {stats['ks_pvalue']:<15.4f} | {stats['wasserstein_distance']:<15.4f} | {similarity_text:<12}")
        
        # Závěr
        print("\nZÁVĚR:")
        if prob_mean > 0.7:
            similarity_level = "velmi podobné"
        elif prob_mean > 0.4:
            similarity_level = "částečně podobné"
        else:
            similarity_level = "málo podobné"
        
        print(f"  Na základě provedené analýzy lze říci, že vygenerované léze jsou {similarity_level}")
        print(f"  reálným lézím z hlediska tvaru.")
        
        # Výpis vlastností s největšími rozdíly
        different_features = [f for f, stats in dist_similarity.items() if not stats['is_similar']]
        if different_features:
            print("\n  Největší rozdíly jsou v následujících vlastnostech:")
            print(f"  {', '.join(different_features)}")
        else:
            print("\n  Nebyly nalezeny statisticky významné rozdíly v žádné vlastnosti.")
        
        print("\nVizualizace byly uloženy do adresáře:", self.output_dir)
        print("="*80)
        
        # Uložení výsledků také jako CSV soubory pro případnou další analýzu
        pd.DataFrame(self.real_features, columns=self.feature_names).to_csv(
            os.path.join(self.output_dir, 'real_features.csv'), index=False)
        
        pd.DataFrame(self.generated_features, columns=self.feature_names).to_csv(
            os.path.join(self.output_dir, 'generated_features.csv'), index=False)
        
        pd.DataFrame({'probability': shape_probs}).to_csv(
            os.path.join(self.output_dir, 'shape_probabilities.csv'), index=False)
        
        pd.DataFrame.from_dict(dist_similarity, orient='index').to_csv(
            os.path.join(self.output_dir, 'distribution_similarity.csv'))


# Příklad použití
def main():
    """Hlavní funkce pro spuštění evaluace"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluace tvarů lézí")
    parser.add_argument("--real_dir", type=str, required=True, help="Adresář s reálnými lézemi")
    parser.add_argument("--gen_dir", type=str, required=True, help="Adresář s vygenerovanými lézemi")
    parser.add_argument("--output_dir", type=str, default="./lesion_evaluation", help="Výstupní adresář pro výsledky")
    parser.add_argument("--num_smallest", type=int, default=20, help="Počet nejmenších reálných lézí pro porovnání")
    
    args = parser.parse_args()
    
    evaluator = LesionShapeEvaluator(
        real_data_dir=args.real_dir,
        generated_data_dir=args.gen_dir,
        output_dir=args.output_dir,
        num_smallest_real=args.num_smallest
    )
    
    # Načtení a zpracování dat
    evaluator.load_and_process_data()
    
    # Generování vizualizací
    evaluator.visualize_results()
    
    # Tisk výsledků místo generování HTML reportu
    evaluator.print_evaluation_results()
    
    print(f"Evaluace dokončena. Výsledky byly vypsány výše a vizualizace uloženy do: {args.output_dir}")


if __name__ == "__main__":
    main()