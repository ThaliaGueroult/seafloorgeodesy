def get_elevation_from_ascii(file_path, target_lon, target_lat):
    with open(file_path, 'r') as file:
        # Lire l'en-tête du fichier ASCII
        ncols = int(file.readline().split()[1])
        nrows = int(file.readline().split()[1])
        xllcorner = float(file.readline().split()[1])
        yllcorner = float(file.readline().split()[1])
        cellsize = float(file.readline().split()[1])
        nodata_value = float(file.readline().split()[1])

        # Calculer la position de la cellule pour la latitude et la longitude données
        col = int((target_lon - xllcorner) / cellsize)
        row = int((target_lat - yllcorner) / cellsize)

        # Vérifier si les coordonnées sont à l'intérieur de la grille
        if col < 0 or col >= ncols or row < 0 or row >= nrows:
            return None

        # Naviguer jusqu'à la ligne correspondante
        for _ in range(row):
            file.readline()

        # Lire la ligne et extraire la valeur d'élévation
        line = file.readline()
        elevation = float(line.split()[col])

        # Vérifier si la valeur est une valeur de non-donnée (NoData)
        if elevation == nodata_value:
            return None

        return elevation

# Test
file_path = "gebco_bermuda.asc"

lat = 31.47356719
lon = 291.28858831 -360 # 291.29859242 - 360
elevation = get_elevation_from_ascii(file_path, lon, lat)
if elevation is not None:
    print(f"L'élévation pour la latitude {lat} et la longitude {lon} est {elevation} mètres.")
else:
    print(f"Les données pour la latitude {lat} et la longitude {lon} ne sont pas disponibles.")
