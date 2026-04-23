// Copyright (c) 2022-2026 Nol Moonen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "build.h"
#include "trace.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>

bool run(
    std::string file_in,
    bool do_render,
    std::string file_out,
    unsigned int size_x,
    unsigned int size_y,
    unsigned int sample_count,
    float3 origin,
    float3 target,
    float3 up) {
    // parse obj file
    scene s;
    RETURN_IF_FALSE(read_scene(s, file_in));

    // build bvh
    bvh bvh;
    RETURN_IF_FALSE(build(s, bvh));

    if (do_render) {
        // generate image
        buf_cpu<uchar> image;
        RETURN_IF_FALSE(image.resize(size_y * size_x * 3));
        RETURN_IF_FALSE(
            generate(size_x, size_y, sample_count, image.get_ptr(), bvh, origin, target, up));

        // write image to file
        stbi_flip_vertically_on_write(1);
        stbi_write_png(file_out.c_str(), size_x, size_y, 3, image.get_ptr(), size_x * 3);
        printf("generated %s\n", file_out.c_str());
    }

    return true;
}

int main(int argc, char *argv[]) {
    std::unordered_map<std::string, std::string> flags{
        {"--out-img", "render.png"}}; // Add flags in here for default values

    std::unordered_set<std::string> accepting_flags{
        "--out-img"}; // Sets which flags expect a value.

    std::vector<std::string> args;

    for (int i = 0; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.length() > 2 && arg.rfind("--", 0) == 0) {
            std::string val;
            if (accepting_flags.find(arg) != accepting_flags.end()) {
                if (i + 1 >= argc) {
                    printf("Missing value for flag %s\n", arg);
                    return EXIT_FAILURE;
                }
                val = argv[i + 1];
                ++i;
            }
            flags[arg] = val;
            continue;
        }
        args.push_back(arg);
    }

    // read input
    bool do_render = flags.find("--render") != flags.end();
    int required_args = (do_render) ? 13 : 1;
    if (args.size() - 1 < required_args) {
        printf("Missing required args. Expected %d but got %d.\n", required_args, args.size() - 1);
        return EXIT_FAILURE;
    }

    unsigned int size_x, size_y, sample_count;
    float3 origin, target, up;

    if (do_render) {
        try {
            size_x = static_cast<unsigned int>(std::stoul(args[2]));
            size_y = static_cast<unsigned int>(std::stoul(args[3]));
            sample_count = static_cast<unsigned int>(std::stoul(args[4]));

            origin.x = std::stof(args[5]);
            origin.y = std::stof(args[6]);
            origin.z = std::stof(args[7]);
            target.x = std::stof(args[8]);
            target.y = std::stof(args[9]);
            target.z = std::stof(args[10]);
            up.x = std::stof(args[11]);
            up.y = std::stof(args[12]);
            up.z = std::stof(args[13]);
        } catch (const std::exception &e) {
            printf("Failed to parse input.\n");
            return EXIT_FAILURE;
        }
    }

    if (!run(args[1], do_render, flags["--out-img"], size_x, size_y, sample_count, origin, target, up)) {
        return EXIT_FAILURE;
    }
}
