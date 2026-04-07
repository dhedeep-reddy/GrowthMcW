#include "solution.h"
#include <algorithm>
#include <fstream>
#include <ios>

// ============================================================
// Optimized Vertical Gaussian Blur (Loop Interchange Applied)
// ============================================================

static void filterVertically(uint8_t *output, const uint8_t *input,
                             const int width, const int height,
                             const int *kernel, const int radius,
                             const int shift)
{
    const int rounding = 1 << (shift - 1);

    // =============================
    // TOP REGION (partial kernel)
    // =============================
    for (int r = 0; r < std::min(radius, height); r++)
    {
        for (int c = 0; c < width; c++)
        {
            int dot = 0;
            int sum = 0;

            auto p = &kernel[radius - r];

            for (int y = 0; y <= std::min(r + radius, height - 1); y++)
            {
                int weight = *p++;
                dot += input[y * width + c] * weight;
                sum += weight;
            }

            int value = static_cast<int>(dot / static_cast<float>(sum) + 0.5f);
            output[r * width + c] = static_cast<uint8_t>(value);
        }
    }

    // =============================
    // MIDDLE REGION (full kernel)
    // =============================
    for (int r = radius; r < height - radius; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int dot = 0;

            const uint8_t* ptr = &input[(r - radius) * width + c];

            for (int i = 0; i < 2 * radius + 1; i++)
            {
                dot += ptr[i * width] * kernel[i];
            }

            int value = (dot + rounding) >> shift;
            output[r * width + c] = static_cast<uint8_t>(value);
        }
    }

    // =============================
    // BOTTOM REGION (partial kernel)
    // =============================
    for (int r = std::max(radius, height - radius); r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            int dot = 0;
            int sum = 0;

            auto p = kernel;

            for (int y = r - radius; y < height; y++)
            {
                int weight = *p++;
                dot += input[y * width + c] * weight;
                sum += weight;
            }

            int value = static_cast<int>(dot / static_cast<float>(sum) + 0.5f);
            output[r * width + c] = static_cast<uint8_t>(value);
        }
    }
}

// ============================================================
// Horizontal Blur (Already Cache Friendly)
// ============================================================

static void filterHorizontally(uint8_t *output, const uint8_t *input,
                               const int width, const int height,
                               const int *kernel, const int radius,
                               const int shift)
{
    const int rounding = 1 << (shift - 1);

    for (int r = 0; r < height; r++)
    {
        // Left border
        for (int c = 0; c < std::min(radius, width); c++)
        {
            int dot = 0;
            int sum = 0;

            auto p = &kernel[radius - c];

            for (int x = 0; x <= std::min(c + radius, width - 1); x++)
            {
                int weight = *p++;
                dot += input[r * width + x] * weight;
                sum += weight;
            }

            int value = static_cast<int>(dot / static_cast<float>(sum) + 0.5f);
            output[r * width + c] = static_cast<uint8_t>(value);
        }

        // Middle (full kernel)
        for (int c = radius; c < width - radius; c++)
        {
            int dot = 0;

            const uint8_t* ptr = &input[r * width + c - radius];

            for (int i = 0; i < 2 * radius + 1; i++)
            {
                dot += ptr[i] * kernel[i];
            }

            int value = (dot + rounding) >> shift;
            output[r * width + c] = static_cast<uint8_t>(value);
        }

        // Right border
        for (int c = std::max(radius, width - radius); c < width; c++)
        {
            int dot = 0;
            int sum = 0;

            auto p = kernel;

            for (int x = c - radius; x < width; x++)
            {
                int weight = *p++;
                dot += input[r * width + x] * weight;
                sum += weight;
            }

            int value = static_cast<int>(dot / static_cast<float>(sum) + 0.5f);
            output[r * width + c] = static_cast<uint8_t>(value);
        }
    }
}

// ============================================================
// 2D Gaussian Blur Wrapper
// ============================================================

void blur(uint8_t *output, const uint8_t *input, const int width,
          const int height, uint8_t *temp)
{
    constexpr int radius = 2;
    constexpr int kernel[radius + 1 + radius] = {1, 4, 6, 4, 1};
    constexpr int shift = 4;

    filterVertically(temp, input, width, height, kernel, radius, shift);
    filterHorizontally(output, temp, width, height, kernel, radius, shift);
}

// ============================================================
// Grayscale Image Loader (PGM P5)
// ============================================================

bool Grayscale::load(const std::string &filename, const int maxSize)
{
    data.reset();

    std::ifstream input(filename.data(),
                        std::ios_base::in | std::ios_base::binary);

    if (!input.is_open())
        return false;

    std::string line;
    input >> line;

    if (line != "P5")
        return false;

    int amplitude;
    input >> width >> height >> amplitude;

    char c;
    input.unsetf(std::ios_base::skipws);
    input >> c;

    if ((width <= 0) || (width > maxSize) ||
        (height <= 0) || (height > maxSize) ||
        (amplitude < 0) || (amplitude > 255) ||
        (c != '\n'))
        return false;

    size = static_cast<size_t>(width) * static_cast<size_t>(height);
    data.reset(new uint8_t[size]);

    if (!data)
        return false;

    input.read(reinterpret_cast<char *>(data.get()), size);

    return !input.fail();
}

// ============================================================
// Grayscale Image Saver
// ============================================================

void Grayscale::save(const std::string &filename)
{
    std::ofstream output(filename.data(),
                         std::ios_base::out | std::ios_base::binary);

    if (!output.is_open())
        return;

    output << "P5\n"
           << width << " " << height << "\n"
           << "255\n";

    if (data)
    {
        output.write(reinterpret_cast<const char *>(data.get()), size);
    }
}