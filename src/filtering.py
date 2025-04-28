import cv2

def applica_soglia_con_gaussiano(percorso_immagine, valore_soglia, kernel_size=(5, 5), sigmaX=0, percorso_output=None):
    """
    Apre un'immagine JPEG, la converte in scala di grigi, applica un filtro gaussiano,
    applica una soglia e azzera i valori di pixel al di sotto della soglia.

    Args:
        percorso_immagine (str): Il percorso completo al file immagine JPEG.
        valore_soglia (int): Il valore di soglia (0-255). I pixel con intensità
                             inferiore a questo valore saranno impostati a 0.
        kernel_size (tuple, optional): Le dimensioni del kernel gaussiano (larghezza, altezza).
                                       Dovrebbero essere numeri interi dispari positivi.
                                       Defaults a (5, 5).
        sigmaX (int, optional): La deviazione standard gaussiana nella direzione X.
                                Se 0, viene calcolata in base alla dimensione del kernel.
                                Defaults a 0.
        percorso_output (str, optional): Il percorso in cui salvare l'immagine
                                         filtrata. Se None, l'immagine filtrata
                                         verrà visualizzata. Defaults a None.
    """
    try:
        # 1. Leggi l'immagine dal disco
        immagine = cv2.imread(percorso_immagine)
        if immagine is None:
            print(f"Errore: Impossibile leggere l'immagine da '{percorso_immagine}'")
            return

        # 2. Converti l'immagine in scala di grigi
        immagine_gray = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)
        #immagine_gray = immagine

        # 3. Applica il filtro gaussiano
        immagine_filtrata = cv2.GaussianBlur(immagine_gray, kernel_size, sigmaX)

        # 4. Applica la soglia
        _, immagine_soglia = cv2.threshold(immagine_filtrata, valore_soglia, 255, cv2.THRESH_BINARY)
        # immagine_soglia = immagine_filtrata

        # immagine_gray = immagine_gray[125:600, 125:600]

        # 5. Visualizza o salva l'immagine filtrata
        if percorso_output:
            cv2.imwrite(percorso_output, immagine_soglia)
            print(f"Immagine filtrata (con Gaussiano) salvata in '{percorso_output}'")
        else:
            cv2.imshow('Immagine Originale (Grayscale)', immagine_gray)
            cv2.imshow('Immagine Filtrata (Gaussiano)', immagine_filtrata)
            cv2.imshow('Immagine Sogliata (con Gaussiano)', immagine_soglia)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    # Esempio di utilizzo:
    percorso_immagine_input = '/root/workspace/repositories/RCF_Pytorch_Updated/out/02668_crop.jpg.png'  # Sostituisci con il percorso della tua immagine
    valore_di_soglia = 110 # Scegli il valore di soglia desiderato
    dimensione_kernel_gaussiano = (21, 21)  # Dimensione del kernel gaussiano (deve essere dispari)
    deviazione_standard_gaussiana_X = 3.0  # Se 0, viene calcolata automaticamente
    # percorso_immagine_output = "/root/workspace/repositories/RCF_Pytorch_Updated/data/test/02668_blurred.jpg"
    percorso_immagine_output = False

    applica_soglia_con_gaussiano(percorso_immagine_input, valore_di_soglia, dimensione_kernel_gaussiano, deviazione_standard_gaussiana_X, percorso_immagine_output)

    # Se non vuoi salvare l'immagine e solo visualizzarla:
    # applica_soglia_con_gaussiano(percorso_immagine_input, valore_di_soglia, dimensione_kernel_gaussiano, deviazione_standard_gaussiana_X)