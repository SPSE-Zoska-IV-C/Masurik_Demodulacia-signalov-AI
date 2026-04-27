# Demodulácia digitálnych rádiových signálov pomocou neurónových sietí

Projekt reprezentuje vizuálne rozhranie v tvare webovej aplikácie, ktorá je postavená na frameworku Dash určená na generovanie, spracovanie a analýzu ASK (Amplitude Shift Keying) signálov s využitím neurónových sietí.

Aplikácia pokrýva workflow:
- generovanie dát
- augmentáciu
- návrh modelu
- tréning
- vyhodnotenie

## Funkcionalita

### Generovanie
- syntetické ASK signály  
- konfigurovateľné parametre (bity, frekvencia, šum, vzorkovanie)

### Augmentácia
- úprava dĺžky signálu (padding alebo resampling)

### Architektúra modelu
- interaktívny návrh vrstiev, ktorý vygeneruje pytorch kód

### Tréning
- režimy: odšumovací alebo predikcia bitov
- sledovanie loss a accuracy  
- ukladanie modelov 

### Vyhodnotenie
- jednotná aj batch evaluácia  
- porovnanie s tradičnou demoduláciou    
- vizualizácie signálov a bitov  


## Technológie
- Python
- GNU Radio
- Dash, Plotly  
- PyTorch  
- NumPy

# License

This project is licensed under the GNU General Public License v3.0.

Copyright (c) 2026

This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, either version 3 of the License, or  
(at your option) any later version.

This program is distributed in the hope that it will be useful,  
but WITHOUT ANY WARRANTY; without even the implied warranty of  
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the  
GNU General Public License for more details.

You should have received a copy of the GNU General Public License  
along with this program. If not, see <https://www.gnu.org/licenses/>.

