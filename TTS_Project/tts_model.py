import torch
import torchaudio
import numpy as np

# Einstellbare Parameter
SAMPLE_RATE = 22050  # Beispiel Sample Rate
SPEED_FACTOR = 1.0   # Geschwindigkeit: 1.0 = normal, < 1.0 = langsamer, > 1.0 = schneller
MAX_TEXT_LENGTH = 256  # Maximale Textlänge, um Eingabedaten auf eine feste Größe zu bringen

# Modellklasse (angepasst an dein Modell)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = torch.nn.Linear(256, 1024)  # Dummy-Modellarchitektur anpassen

    def forward(self, x):
        return self.layer(x)

# Nachverarbeitung: Modell-Ausgabe in Audio umwandeln und Geschwindigkeit anpassen
def postprocess_output(output, sample_rate=SAMPLE_RATE, speed_factor=SPEED_FACTOR):
    # Umwandlung der Modell-Ausgabe in ein numpy-Array
    output = output.squeeze().detach().numpy()
    
    # Resampling, um die Geschwindigkeit zu ändern
    if speed_factor != 1.0:
        # Berechne die neue Samplerate basierend auf der Geschwindigkeit
        new_sample_rate = sample_rate * speed_factor
        
        # Resample das Audio (Verwende torchaudio.transforms.Resample)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=int(new_sample_rate))
        output = torch.tensor(output).unsqueeze(0)  # Umwandlung in Tensor für Resampling
        output = resampler(output).squeeze().numpy()
        
    return output.astype(np.float32), int(sample_rate * speed_factor)

# Text-Vorverarbeitung
def preprocess_text(text):
    encoded_text = [ord(char) for char in text]  # Einfache ASCII-Encoding (Dummy-Beispiel)
    tensor = torch.tensor(encoded_text, dtype=torch.float32).unsqueeze(0)  # Umwandlung in FloatTensor
    
    # Sicherstellen, dass die Eingabe auf die Größe (1, 256) umgeformt wird
    # Text auf MAX_TEXT_LENGTH begrenzen und ggf. mit 0 (Padding) auffüllen
    tensor = tensor.view(1, -1)  # Ungeformte Dimensionen

    # Wenn der Text kürzer als MAX_TEXT_LENGTH ist, fügen wir Nullen hinzu (Padding)
    if tensor.shape[1] < MAX_TEXT_LENGTH:
        padding = torch.zeros(1, MAX_TEXT_LENGTH - tensor.shape[1], dtype=torch.float32)
        tensor = torch.cat([tensor, padding], dim=1)
    # Wenn der Text länger ist, wird er abgeschnitten
    elif tensor.shape[1] > MAX_TEXT_LENGTH:
        tensor = tensor[:, :MAX_TEXT_LENGTH]
    
    return tensor

# Text-to-Speech-Funktion
def text_to_speech(model, text, output_path="output.wav", sample_rate=SAMPLE_RATE, speed_factor=SPEED_FACTOR):
    input_tensor = preprocess_text(text)  # Text in Modell-Eingabeformat umwandeln
    with torch.no_grad():  # Kein Gradient erforderlich
        output = model(input_tensor)  # Modell ausführen
    audio, sr = postprocess_output(output, sample_rate=sample_rate, speed_factor=speed_factor)
    torchaudio.save(output_path, torch.tensor([audio]), sample_rate=sr)
    print(f"Audio wurde als {output_path} gespeichert!")

# Modell laden
def load_model(model_path):
    # Lade die Checkpoint-Datei
    checkpoint = torch.load(model_path)

    # Prüfe, ob ein state_dict vorliegt
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Initialisiere das Modell
    model = MyModel()

    # Filter state_dict: Entferne unerwartete Schlüssel
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if k in model.state_dict()
    }

    # Gewichte ins Modell laden
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()  # In den Evaluationsmodus versetzen
    return model

# Hauptprogramm
if __name__ == "__main__":
    print("Bitte gib deinen Text ein, der umgewandelt werden soll: ")
    user_text = input("")

    try:
        # Modell laden
        model_path = "GRSoraKH2_e200_s27800/GRSoraKH2.pth"  # Beispiel für den Pfad
        model = load_model(model_path)
        
        # Geschwindigkeit anpassen (z.B. 0.5 für langsamer, 2.0 für schneller)
        speed_factor = float(input("Gib einen Wert für die Geschwindigkeit ein (z.B. 1.0 für normal): "))

        # Text-to-Speech ausführen
        text_to_speech(model, user_text, speed_factor=speed_factor)
        
    except Exception as e:
        print(f"Fehler: {e}")
