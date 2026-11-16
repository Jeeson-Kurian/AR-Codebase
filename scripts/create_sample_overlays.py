"""Create eleven transparent calibration overlays for the webcam examples."""
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    directory = Path("hairstyles")
    directory.mkdir(exist_ok=True)
    for number in range(1, 12):
        path = directory / f"hairstyle{number}.png"
        image = Image.new("RGBA", (240, 120))
        draw = ImageDraw.Draw(image)
        colour = (40 + 15 * number, 180 - 8 * number, 230 - 10 * number, 230)
        # A crown arc and centre mark make scale, tilt and horizontal drift visible.
        draw.arc((10, 12, 230, 166), 180, 360, fill=colour, width=10)
        draw.line((120, 4, 120, 36), fill=colour, width=4)
        draw.text((130, 15), str(number), fill=colour)
        try:
            with path.open("xb") as dest:
                image.save(dest, format="PNG")
        except FileExistsError:
            print(f"Keeping existing {path}")


if __name__ == "__main__":
    main()
