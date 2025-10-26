import gzip
import shutil
from pathlib import Path


source_folder = Path(
    "../Data/images_zip/Outputs/cpac/nofilt_noglobal/reho"
)  # Folder with .gz files
destination_folder = Path(
    "../Data/images_zip/Outputs/cpac/nofilt_noglobal/extracted_reho"
)  # Folder for extracted files

destination_folder.mkdir(parents=True, exist_ok=True)

for gz_file in source_folder.glob("*.gz"):
    output_file = destination_folder / gz_file.stem
    try:
        with gzip.open(gz_file, "rb") as f_in, open(output_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Extracted: {gz_file.name} → {output_file.name}")
    except (gzip.BadGzipFile, EOFError) as e:
        print(f"Skipped {gz_file.name}: not a valid gzip file or corrupted ({e})")

print("All files extracted successfully.")
