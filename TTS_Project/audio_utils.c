#include <stdio.h>
#include <stdlib.h>

void slow_down_audio(float* audio, int length, int factor) {
    // Hier implementierst du eine einfache Methode zur Verlangsamung des Audios
    int new_length = length * factor;
    float* slowed_audio = (float*)malloc(new_length * sizeof(float));

    for (int i = 0; i < new_length; i++) {
        int idx = i / factor;
        if (idx < length) {
            slowed_audio[i] = audio[idx];
        } else {
            slowed_audio[i] = 0;  // Einfache Interpolation für das Beispiel
        }
    }

    // Audio wieder an den Aufrufer zurückgeben oder in Datei speichern
    for (int i = 0; i < new_length; i++) {
        printf("%f\n", slowed_audio[i]);  // Hier einfaches Drucken (dies würde an eine Datei weitergegeben)
    }

    free(slowed_audio);
}
