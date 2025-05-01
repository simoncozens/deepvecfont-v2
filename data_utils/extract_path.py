import uharfbuzz as hb
from fontTools.pens.qu2cuPen import Qu2CuPen
from fontTools.pens.svgPathPen import SVGPathPen, pointToString


class AbsoluteSVGPathPen(SVGPathPen):
    def _lineTo(self, pt):
        t = "L"
        t += pointToString(pt, self._ntos)
        self._commands.append(t)

    def _closePath(self):
        pass


def extract_path(font, char, upem, variant=None):
    if variant:
        font.set_variations(variant)
    glyph = font.get_nominal_glyph(ord(char))
    path = []
    if glyph is None:
        return path
    svgpen = AbsoluteSVGPathPen({})
    pen = Qu2CuPen(svgpen, max_err=5, all_cubic=True)
    font.draw_glyph_with_pen(glyph, pen)
    for command in svgpen._commands:
        if command[0] == " ":
            command = "L" + command[1:]
        elements = [command[0]] + [float(x) for x in command[1:].split()]
        # Y coordinates need to be inverted for SVG; subtract from upem
        if len(elements) > 2:
            elements[2] = upem - elements[2]
        if len(elements) > 4:
            elements[4] = upem - elements[4]
        if len(elements) > 6:
            elements[6] = upem - elements[6]

        path.append(elements)
    return path


def make_hb_font(filepath):
    blob = hb.Blob.from_file_path(filepath)
    face = hb.Face(blob)
    font = hb.Font(face)
    return font, face.upem


if __name__ == "__main__":
    font, upem = make_hb_font("./font_ttfs/train/00001.ttf")
    print(extract_path(font, "S", upem))
