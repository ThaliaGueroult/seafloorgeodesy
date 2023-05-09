import re

def extract_lat_lon(filename):
    with open(filename, 'r') as f:
        content = f.read()
        lat_match = re.search(r'NMEA Latitude = (\d+) (\d+\.\d+) [NS]', content)
        lat_degrees = float(lat_match.group(1))
        lat_minutes = float(lat_match.group(2))
        lat_decimal = lat_degrees + lat_minutes / 60.0
        if 'S' in lat_match.group():
            lat_decimal *= -1.0

        lon_match = re.search(r'NMEA Longitude = (\d+) (\d+\.\d+) [WE]', content)
        lon_degrees = float(lon_match.group(1))
        lon_minutes = float(lon_match.group(2))
        lon_decimal = lon_degrees + lon_minutes / 60.0
        if 'W' in lon_match.group():
            lon_decimal *= -1.0

        return lat_decimal, lon_decimal