import math
import os
from http import HTTPStatus
from typing import List

import cv2
import numpy as np
import requests

MAPBOX_TOKEN = os.environ['MAPBOX_TOKEN']

class TileDownloader:
    __MAPBOX_API_URL = "https://api.mapbox.com/v4/{0}/{3}/{1}/{2},512@2x.pngraw?access_token={4}"
    __MAPBOX_TILE_SIZE = 512
    __MAPBOX_TILESET_HEIGHTFIELD = "mapbox.mapbox-terrain-dem-v1"
    __MAPBOX_TILESET_BASEMAP = "mapbox.satellite"
    __MAPBOX_TILESET_WATER = "mapbox.mapbox-streets-v8"

    def __del__( self ):
        self.__tile_count = None
        self.__tile_start_x = None
        self.__tile_start_y = None
        self.__zoom_level = None

    def __init__( self, texture_size: int, longitude_min: float, longitude_max: float, latitude: float ):
        zoom_level = self.__calculate_zoom_level( texture_size, longitude_min, longitude_max )
        self.__tile_count = self.__calculate_tile_count( zoom_level, longitude_min, longitude_max )
        self.__tile_start_x = self.__calculate_tile_x( zoom_level, longitude_min )
        self.__tile_start_y = self.__calculate_tile_y( zoom_level, latitude )
        self.__zoom_level = zoom_level

    def run( self, output_folder: str, output_filename: str, force_download: bool = False ):
        print( f'\n  o Processing "{output_filename}"' )
        self.__download_tiles(
            output_folder,
            f"{output_filename}_basemap",
            self.__MAPBOX_TILESET_BASEMAP,
            force_download=force_download,
            process_heightfield=False,
        )
        self.__download_tiles(
            output_folder,
            f"{output_filename}_heightfield",
            self.__MAPBOX_TILESET_HEIGHTFIELD,
            force_download=force_download,
            process_heightfield=True,
        )
        self.__download_tiles(
            output_folder,
            f"{output_filename}_water",
            self.__MAPBOX_TILESET_WATER,
            force_download=force_download,
            process_heightfield=False,
        )

    def __download_tiles(
        self,
        output_folder: str,
        output_filename: str,
        tileset_id: str,
        force_download: bool,
        process_heightfield: bool,
    ):
        tile_folder = self.__get_tile_folder( output_folder )
        os.makedirs( tile_folder, exist_ok=True )
        if os.path.isdir( tile_folder ):
            tile_paths = []
            for y_index in range( self.__tile_count ):
                tile_y = self.__tile_start_y + y_index
                for x_index in range( self.__tile_count ):
                    tile_x = self.__tile_start_x + x_index
                    tile_path = self.__get_tile_path( output_folder, tile_x, tile_y, tileset_id )
                    tile_paths.append( tile_path )
                    if force_download or not os.path.exists( tile_path ):
                        print(f"    - Downloading Tile ({tile_x}, {tile_y}) ({tileset_id})")
                        url = self.__MAPBOX_API_URL.format( tileset_id, tile_x, tile_y, self.__zoom_level, MAPBOX_TOKEN )
                        response = requests.get( url )
                        if response and response.status_code == HTTPStatus.OK:
                            with open(tile_path, "wb") as file:
                                file.write( response.content )
            output_image = self.__stitch_images( tile_paths, self.__tile_count, self.__tile_count )
            if process_heightfield:
                output_image = self.__convert_to_heightmap( output_image )
                value_min = np.min( output_image )
                value_max = np.max( output_image )
                value_range = max( 0.0, value_max - value_min )
                if value_range > 0.0:
                    output_image = ( output_image - value_min ) / value_range
                output_image = np.uint16( output_image * 65535.0 )
                print(f"     - {value_range} ({value_min}, {value_max})")
            output_path = os.path.abspath( os.path.join( output_folder, f"{output_filename}.png" ) )
            cv2.imwrite( output_path, output_image )

    @classmethod
    def __calculate_tile_count( cls, zoom_level: int, longitude_min: float, longitude_max: float ) -> int:
        tile_x_max = cls.__calculate_tile_x( zoom_level, longitude_max )
        tile_x_min = cls.__calculate_tile_x( zoom_level, longitude_min )
        tile_count = max( 1, tile_x_max - tile_x_min + 1 )
        return int( pow( 2, int( math.ceil( math.log2( tile_count ) ) ) ) )

    @classmethod
    def __calculate_tile_x( cls, zoom_level: int, longitude: float ) -> int:
        return int( pow( 2, zoom_level ) * ( ( longitude + 180.0 ) / 360.0 ) )

    @classmethod
    def __calculate_tile_y( cls, zoom_level: int, latitude: float ) -> int:
        radians = math.pi * latitude / 180.0
        factor = pow( 2, zoom_level )
        return int( factor * ( 1.0 - ( math.log( math.tan( radians ) + ( 1.0 / math.cos( radians ) ) ) / math.pi ) ) / 2.0 )

    @classmethod
    def __calculate_zoom_level( cls, texture_size: int, longitude_min: float, longitude_max: float ) -> int:
        pixels = cls.__MAPBOX_TILE_SIZE * ( ( longitude_max - longitude_min ) / 360.0 )
        factor = max( 1.0, float( texture_size ) / pixels )
        return int( math.log2( factor ) )

    @classmethod
    def __convert_to_heightmap( cls, array ) -> np.array:
        R = array[:, :, 0]
        G = array[:, :, 1]
        B = array[:, :, 2]
        return -10000 + ( ( R * 256 * 256 + G * 256 + B ) * 0.1 )

    @classmethod
    def __get_tile_folder( cls, output_folder: str ) -> str:
        return os.path.abspath( os.path.join( output_folder, "download" ) )

    @classmethod
    def __get_tile_path( cls, output_folder: str, tile_x: int, tile_y: int, tileset_id: str ) -> str:
        return os.path.join( cls.__get_tile_folder( output_folder ), f"{tile_x}_{tile_y}_{tileset_id}.png" )

    @classmethod
    def __stitch_images( cls, image_paths: List[str], x_count: int, y_count: int ) -> np.array:
        stitched_image = None
        if len( image_paths ) == x_count * y_count:
            first_image = cv2.imread( image_paths[0] )
            image_height, image_width = first_image.shape[:2]
            stitched_width = image_width * x_count
            stitched_height = image_height * y_count
            stitched_image = np.zeros( ( stitched_height, stitched_width, 3 ), dtype=np.uint8 )
            for i, path in enumerate( image_paths ):
                image = cv2.imread( path )
                x_offset = ( i % x_count ) * image_width
                y_offset = ( i // x_count ) * image_height
                stitched_image[y_offset : y_offset + image_height, x_offset : x_offset + image_width] = image
        return stitched_image


def download_tiles(
    texture_size: int,
    longitude_min: float,
    longitude_max: float,
    latitude: float,
    output_folder: str,
    output_filename: str,
):
    tile_downloader = TileDownloader( texture_size, longitude_min, longitude_max, latitude )
    tile_downloader.run( output_folder, output_filename )


def download_milford_sound():
    download_tiles(
        texture_size=4096,
        longitude_min=167.77,
        longitude_max=168.0,
        latitude=-44.57,
        output_folder="C:\\Temp\\MapBox\\milford_sound",
        output_filename="milford_sound",
    )


if __name__ == "__main__":
    download_milford_sound()