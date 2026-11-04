ENV_NAME = meta-netze

.PHONY: create update update-conda update-pip clean help

# Neue Umgebung erstellen
create:
	conda env create -f environment.yml

# Alles aktualisieren
update: update-conda update-pip

# Nur Conda-Pakete aktualisieren
update-conda:
	conda env update -f environment.yml --prune

# Nur Pip-Pakete aktualisieren
update-pip:
	pip install -r requirements.txt

# Umgebung löschen
clean:
	conda env remove -n $(ENV_NAME)

# Hilfe anzeigen
help:
	@echo "Available commands | Verfügbare Befehle"
	@echo "  make create       - Create new conda environment  | Neue Conda-Umgebung erstellen"
	@echo "  make update       - Update conda and pip packages | Conda- und Pip-Pakete aktualisieren"
	@echo "  make update-conda - Update only conda packages    | Nur Conda-Pakete aktualisieren"
	@echo "  make update-pip   - Update only pip packages      | Nur Pip-Pakete aktualisieren"
	@echo "  make clean        - Remove conda environment      | Conda-Umgebung entfernen"