BANANA_ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "4:3": (1184, 864),
    "3:4": (864, 1184),
    "2:3": (832, 1248),
    "3:2": (1248, 832),
    "9:16": (768, 1344),
    "16:9": (1344, 768),
}

QWEN_IMAGE_ASPECT_RATIOS = {
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
}

BANANA_PRO_ASPECT_RATIOS = {
    "1K": {
        "1:1": (1024, 1024),
        "4:3": (1200, 896),
        "3:4": (896, 1200),
        "2:3": (848, 1264),
        "3:2": (1264, 848),
        "9:16": (768, 1376),
        "16:9": (1376, 768),
        "4:5": (928, 1152),
        "5:4": (1152, 928),
        "21:9": (1584, 672),
    },
    "2K": {
        "1:1": (2048, 2048),
        "4:3": (2400, 1792),
        "3:4": (1792, 2400),
        "2:3": (1696, 2528),
        "3:2": (2528, 1696),
        "9:16": (1536, 2752),
        "16:9": (2752, 1536),
        "4:5": (1856, 2304),
        "5:4": (2304, 1856),
        "21:9": (3168, 1344),
    },
    "4K": {
        "1:1": (4096, 4096),
        "4:3": (4800, 3584),
        "3:4": (3584, 4800),
        "2:3": (3392, 5056),
        "3:2": (5056, 3392),
        "9:16": (3072, 5504),
        "16:9": (5504, 3072),
        "4:5": (3712, 4608),
        "5:4": (4608, 3712),
        "21:9": (6336, 2688),
    },
}

VIDEO_RESOLUTION_SHORT_SIDE = {
    "480P": 480,
    "720P": 720,
    "1080P": 1080,
}

VIDEO_ASPECT_RATIO_LABELS = tuple(BANANA_PRO_ASPECT_RATIOS["1K"].keys())


class aspect_ratio_matcher:
    """
    ComfyUI node:
    Match the closest preset aspect ratio from the input image and
    return the corresponding target width and height.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": ("COMBO", {
                    "default": "banana",
                    "options": ["banana", "banana_pro", "qwen_image", "video"],
                    "tooltip": "选择分辨率映射模式。video 模式用于 480P / 720P / 1080P 这类视频规格。"
                }),
                "resolution": ("COMBO", {
                    "default": "720P",
                    "options": ["1K", "2K", "4K", "480P", "720P", "1080P"],
                    "tooltip": "统一分辨率选项：banana_pro 使用 1K/2K/4K，video 使用 480P/720P/1080P，其它 mode 会忽略此项。"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("aspect_ratio", "width", "height")
    FUNCTION = "match_aspect_ratio"
    CATEGORY = "AFL/Image Calculator"

    @staticmethod
    def calculate_aspect_ratio(width, height):
        if height == 0:
            return 0.0
        return width / height

    @staticmethod
    def parse_aspect_ratio(ratio_str):
        width_str, height_str = ratio_str.split(":")
        return int(width_str), int(height_str)

    @staticmethod
    def round_to_even(value):
        rounded = int(round(value))
        return rounded if rounded % 2 == 0 else rounded + 1

    def find_closest_aspect_ratio(self, input_ratio, aspect_ratios):
        ratio_values = {key: width / height for key, (width, height) in aspect_ratios.items()}

        closest_ratio = None
        min_difference = float("inf")

        for ratio_str, ratio_value in ratio_values.items():
            difference = abs(input_ratio - ratio_value)
            if difference < min_difference:
                min_difference = difference
                closest_ratio = ratio_str

        return closest_ratio

    def build_video_aspect_ratios(self, video_resolution):
        short_side = VIDEO_RESOLUTION_SHORT_SIDE[video_resolution]
        aspect_ratios = {}

        for ratio_str in VIDEO_ASPECT_RATIO_LABELS:
            ratio_width, ratio_height = self.parse_aspect_ratio(ratio_str)

            if ratio_width >= ratio_height:
                height = short_side
                width = self.round_to_even(short_side * ratio_width / ratio_height)
            else:
                width = short_side
                height = self.round_to_even(short_side * ratio_height / ratio_width)

            aspect_ratios[ratio_str] = (width, height)

        return aspect_ratios

    def match_aspect_ratio(
        self,
        image,
        mode,
        resolution="720P",
    ):
        if mode == "banana":
            aspect_ratios = BANANA_ASPECT_RATIOS
        elif mode == "banana_pro":
            banana_pro_resolution = resolution if resolution in BANANA_PRO_ASPECT_RATIOS else "1K"
            aspect_ratios = BANANA_PRO_ASPECT_RATIOS[banana_pro_resolution]
        elif mode == "video":
            video_resolution = resolution if resolution in VIDEO_RESOLUTION_SHORT_SIDE else "720P"
            aspect_ratios = self.build_video_aspect_ratios(video_resolution)
        else:
            aspect_ratios = QWEN_IMAGE_ASPECT_RATIOS

        _, height, width, _ = image.shape
        input_ratio = self.calculate_aspect_ratio(width, height)

        closest_ratio = self.find_closest_aspect_ratio(input_ratio, aspect_ratios)
        target_width, target_height = aspect_ratios[closest_ratio]

        return (closest_ratio, target_width, target_height)


NODE_CLASS_MAPPINGS = {
    "AFL:aspect_ratio_matcher": aspect_ratio_matcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:aspect_ratio_matcher": "Aspect Ratio Matcher"
}
