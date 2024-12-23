#include <iostream>
#include <fstream>

constexpr auto PRINT_EVERY = 512;

inline std::string get_printout_path(const std::string &filename)
{
    return "/workspace/blocksparse-dit/csrc/matmul_a100_cute/printouts/" + filename + ".txt";
}
template <typename T>
void read_printout(const std::string &filename, T *tensor, int M, int N)
{
    std::cout << "Reading printout " << filename << std::endl;
    std::ifstream file(get_printout_path(filename));
    if (!file.is_open())
    {
        throw std::runtime_error("Error: File " + get_printout_path(filename) + " does not exist.");
    }
    for (int i = 0; i < M; ++i)
    {
        if (i % PRINT_EVERY == 0 and i > 0)
        {
            std::cout << "... " << i << " / " << M << std::endl;
        }
        for (int j = 0; j < N; ++j)
        {
            float in;
            file >> in;
            tensor[i * N + j] = T(in);
        }
    }
}
template <>
void read_printout(const std::string &filename, uint8_t *tensor, int M, int N)
{
    std::cout << "Reading printout " << filename << std::endl;
    std::ifstream file(get_printout_path(filename));
    if (!file.is_open())
    {
        throw std::runtime_error("Error: File " + get_printout_path(filename) + " does not exist.");
    }
    int write_loc = 0;
    for (int i = 0; i < M; ++i)
    {
        if (i % PRINT_EVERY == 0 and i > 0)
        {
            std::cout << "... " << i << " / " << M << std::endl;
        }
        for (int j = 0; j < N; ++j)
        {
            float in;
            file >> in;
            tensor[write_loc++] = uint8_t(in);
            tensor[write_loc++] = 0xFF;
            tensor[write_loc++] = 0xFF;
            tensor[write_loc++] = 0xFF;
        }
    }
}

template <>
void read_printout<bool>(const std::string &filename, bool *tensor, int M, int N)
{
    std::cout << "Reading printout " << filename << std::endl;
    std::ifstream file(get_printout_path(filename));
    if (!file.is_open())
    {
        throw std::runtime_error("Error: File " + get_printout_path(filename) + " does not exist.");
    }
    bool print_all = M < 32 && N < 32;
    for (int i = 0; i < M; ++i)
    {
        if (i % PRINT_EVERY == 0 and i > 0)
        {
            std::cout << "... " << i << " / " << M << std::endl;
        }
        for (int j = 0; j < N; ++j)
        {
            bool in;
            file >> in;
            tensor[i * N + j] = in;
            if (print_all)
                std::cout << in << " ";
        }
        if (print_all)
            std::cout << std::endl;
    }
}