#include "LBlock.h"

namespace std {
    template<>
    struct hash<miniTemplate> {
        std::size_t operator()(const miniTemplate& m) const {
            return std::hash<unsigned long long>()(m.to_ullong());
        }
    };
}

int hammingWeight(const miniTemplate& t) {
    return static_cast<int>(t.count());
}

miniTemplate Rho(const miniTemplate& input) {
    miniTemplate result;
    for (int i = 0; i < 8; ++i)
        result[i] = input[(i + 1) % 8];     
    return result;
}

miniTemplate RhoInverse(const miniTemplate& input) {
    miniTemplate result;
    for (int i = 0; i < 8; ++i)
        result[(i + 1) % 8] = input[i]; 
    return result;
}

miniTemplate P(const miniTemplate& input) {
    miniTemplate result;
    result[7] = input[6]; // U7 = Z6
    result[6] = input[4]; // U6 = Z4
    result[5] = input[7]; // U5 = Z7
    result[4] = input[5]; // U4 = Z5
    result[3] = input[2]; // U3 = Z2
    result[2] = input[0]; // U2 = Z0
    result[1] = input[3]; // U1 = Z3
    result[0] = input[1]; // U0 = Z1
    return result;
}

std::set<std::string> computePhiSystem(const Template& a_template, const Template& b_template) {
    std::vector<int> uncertain_indices;
    std::bitset<num_blocks> base_result;

    for (int i = 0; i < num_blocks; ++i) {
        bool ai = a_template[i];
        bool bi = b_template[i];

        if (!ai && !bi) {
            base_result[i] = 0;
        }
        else if ((ai == 1 && bi == 0) || (ai == 0 && bi == 1)) {
            base_result[i] = 1;
        }
        else if (ai == 1 && bi == 1) {
            uncertain_indices.push_back(i);
        }
    }

    std::set<std::string> result;
    int total_variants = 1 << uncertain_indices.size();

    for (int mask = 0; mask < total_variants; ++mask) {
        std::bitset<num_blocks> c = base_result;
        for (size_t j = 0; j < uncertain_indices.size(); ++j) {
            int index = uncertain_indices[j];
            c[index] = (mask >> j) & 1;
        }
        result.insert(c.to_string());
    }

    return result;
}

void precomputePhiSystemToFile(const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << " Failed to open file: " << filename << "\n";
        return;
    }


    for (TemplateID a_val = 0; a_val < (1 << num_blocks); ++a_val) {
        Template a(a_val);

        for (TemplateID b_val = 0; b_val < (1 << num_blocks); ++b_val) {
            Template b(b_val);

            std::set<std::string> phi_set = computePhiSystem(a, b);
            TemplateID line_id = (a_val << 8) + b_val; // a * 256 + b

            //file << line_id << ":";

            for (const auto& phi_str : phi_set) {
                Template phi(phi_str);
                TemplateID phi_val = phi.to_ulong();
                file << " " << phi_val;
            }
            file << "\n";
        }
    }

    file.close();
    std::cout << "Phi system written to " << filename << "\n";
}

void loadMiniPhiSystem(const std::string& filename, std::vector<std::unordered_set<miniTemplate>>& Phi_system) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open Phi_system.txt\n";
        return;
    }

    constexpr int phi_size = 1 << 16;
    Phi_system.clear();
    Phi_system.resize(phi_size);

    std::string line;
    int line_id = 0;

    while (std::getline(file, line) && line_id < phi_size) {
        std::istringstream iss(line);
        TemplateID phi_val;
        while (iss >> phi_val) {
            Phi_system[line_id].insert(miniTemplate(phi_val));
        }
        ++line_id;
    }

    if (line_id != phi_size) {
        std::cerr << "Warning: expected 256 lines but got " << line_id << "\n";
    }

    file.close();
    std::cout << "Mini Phi system loaded into memory (" << line_id << " entries).\n";
}

const uint8_t LBlock_SBOX[16] = {
    14, 9, 15, 0,
    13, 4, 10, 11,
    1, 2, 8, 3,
    7, 6, 12, 5
};

std::map<double, size_t> u_dist;
std::vector<std::vector<double>> partial_sums;

std::array<std::array<int, 16>, 16> computeDDT() {
    std::array<std::array<int, 16>, 16> DDT = {};

    for (int alpha = 0; alpha < 16; ++alpha) {
        for (int x = 0; x < 16; ++x) {
            int y = LBlock_SBOX[x ^ alpha] ^ LBlock_SBOX[x];
            DDT[alpha][y]++;
        }
    }

    return DDT;
}

std::map<double, size_t> computeU1(const std::array<std::array<int, 16>, 16>& DDT) {
    std::map<double, size_t> u1;

    for (int alpha = 1; alpha < 16; ++alpha) { 
        for (int beta = 0; beta < 16; ++beta) {
            int count = DDT[alpha][beta];
            if (count > 0) {
                double prob = static_cast<double>(count) / 16.0; // 4-bit S-box
                u1[prob]++;
            }
        }
    }

    return u1;
}

std::map<double, size_t> convolveDist(const std::map<double, size_t>& uA, const std::map<double, size_t>& u1) {
    std::map<double, size_t> result;

    for (const std::pair<const double, size_t>& entry1 : uA) {
        double p1 = entry1.first;
        size_t n1 = entry1.second;

        for (const std::pair<const double, size_t>& entry2 : u1) {
            double p2 = entry2.first;
            size_t n2 = entry2.second;

            double prod = p1 * p2;
            result[prod] += n1 * n2;
        }
    }

    return result;
}

std::vector<std::map<double, size_t>> compute_u_distributions_grouped(int max_A) {
    auto DDT = computeDDT();
    auto u1_map = computeU1(DDT);

    std::vector<std::map<double, size_t>> u_maps;
    u_maps.push_back(u1_map); // u1

    for (int A = 2; A <= max_A; ++A) {
        auto uA = convolveDist(u_maps[A - 2], u1_map);
        u_maps.push_back(uA);
        std::cout << "Computed grouped u[" << A << "] (unique probabilities = " << uA.size() << ")\n";
    }


    return u_maps;
}

void save_u_distributions_to_file(const std::vector<std::map<double, size_t>>& u_maps, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return;
    }

    for (size_t A = 0; A < u_maps.size(); ++A) {
        out << "A = " << A + 1 << "\n";
        for (const auto& entry : u_maps[A]) {
            out << std::setprecision(16) << entry.first << "," << entry.second << "\n";
        }
        out << "\n";
    }

    out.close();
    std::cout << "u_i(A) distributions saved to: " << filename << "\n";
}

std::vector<std::map<double, size_t>> load_u_distributions_from_file(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::map<double, size_t>> u_maps;
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return u_maps;
    }

    std::string line;
    std::map<double, size_t> current_map;

    while (std::getline(file, line)) {
        if (line.empty()) {
            if (!current_map.empty()) {
                u_maps.push_back(current_map);
                current_map.clear();
            }
            continue;
        }

        if (line.find("A = ") == 0)
            continue;

        std::istringstream iss(line);
        double prob;
        size_t count;
        char comma;
        if (iss >> prob >> comma >> count) {
            current_map[prob] = count;
        }
    }

    if (!current_map.empty()) {
        u_maps.push_back(current_map);
    }

    file.close();
    std::cout << "Loaded u_i(A) distributions from: " << filename << "\n";
    return u_maps;
}

double sum_u_i_limited(int w, size_t limit, const std::vector<std::map<double, size_t>>& u_distributions) {
    if (w <= 0 || w > static_cast<int>(u_distributions.size())) return 0.0;

    const auto& u = u_distributions[w - 1]; // A = w
    double sum = 0.0;
    size_t count = 0;

    for (const auto& entry : u) {
        double prob = entry.first;
        size_t freq = entry.second;

        for (size_t i = 0; i < freq && count < limit; ++i, ++count) {
            sum += prob;
        }

        if (count >= limit)
            break;
    }

    return sum;
}

std::vector<std::vector<double>> precompute_partial_sums(
    const std::vector<std::map<double, size_t>>& u_distributions,
    size_t max_limit_per_A = 1000000
) {
    std::vector<std::vector<double>> partial_sums;

    for (const auto& u : u_distributions) {
        std::vector<double> sums;
        sums.reserve(max_limit_per_A);
        size_t count = 0;
        double current_sum = 0.0;

        for (const auto& entry : u) {
            double prob = entry.first;
            size_t freq = entry.second;

            for (size_t i = 0; i < freq && count < max_limit_per_A; ++i, ++count) {
                current_sum += prob;
                sums.push_back(current_sum);
            }

            if (count >= max_limit_per_A)
                break;
        }

        partial_sums.push_back(sums); // sums[i] = sum_{j=0}^{i} u_j
    }

    return partial_sums;
}

double get_precomputed_sum(
    int w, size_t limit,
    const std::vector<std::vector<double>>& partial_sums
) {
    if (w <= 0 || w > static_cast<int>(partial_sums.size())) return 0.0;
    const auto& vec = partial_sums[w - 1];
    if (limit == 0) return 0.0;
    if (limit >= vec.size()) return vec.back();
    return vec[limit - 1];
}

void computeUB(
    const std::vector<std::unordered_set<miniTemplate>>& Phi_system,
    int rounds,
    const std::string& base_filename,
    double p,
    std::ofstream& log,
    const std::vector<std::map<double, size_t>>& u_distributions,
    const std::vector<std::vector<double>>& precomputed_sums) {

    const size_t dim = 1 << num_blocks;
    const std::string round_file = base_filename + "_r";

    // --- Round 2 ---
    std::string file_r = round_file + "2.bin";
    std::ofstream out_r(file_r, std::ios::binary);
    if (!out_r.is_open()) {
        std::cerr << "Cannot open file for round 2: " << file_r << "\n";
        return;
    }

    double max_t2 = 0.0;
    TemplateID a_max2 = 0, b_max2 = 0;

    auto start_time = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic)
    for (TemplateID a_val = 0; a_val < dim; ++a_val) {
        std::vector<double> row(dim, 0.0);
        Template a(a_val);
        miniTemplate a_x, a_y;
        for (int i = 0; i < num_blocks / 2; ++i) a_x[i] = a[i];
        for (int i = 0; i < num_blocks / 2; ++i) a_y[i] = a[i + num_blocks / 2];

        miniTemplate Pa_y = P(a_y);
        miniTemplate rho_ax = Rho(a_x);
        miniTemplate rho_ay = Rho(a_y);

        for (TemplateID b_val = 0; b_val < dim; ++b_val) {
            Template b(b_val);
            miniTemplate b_x, b_y;
            for (int i = 0; i < num_blocks / 2; ++i) b_x[i] = b[i];
            for (int i = 0; i < num_blocks / 2; ++i) b_y[i] = b[i + num_blocks / 2];

            TemplateID line1 = (Pa_y.to_ullong() << 4) + rho_ax.to_ullong();
            TemplateID line2 = (b_x.to_ullong() << 4) + rho_ay.to_ullong();

            bool cond1 = (line1 < Phi_system.size() &&
                Phi_system[line1].find(b_x) != Phi_system[line1].end());
            bool cond2 = (line2 < Phi_system.size() &&
                Phi_system[line2].find(b_y) != Phi_system[line2].end());

            if (cond1 && cond2) {
                uint8_t wt = hammingWeight(a_y) + hammingWeight(b_x);
                row[b_val] = std::pow(p, wt);

#pragma omp critical
                if (a_val != 0 && row[b_val] > max_t2) {
                    max_t2 = row[b_val];
                    a_max2 = a_val;
                    b_max2 = b_val;
                }
            }
        }

#pragma omp critical
        {
            out_r.seekp(static_cast<std::streamoff>(a_val) * dim * sizeof(double));
            out_r.write(reinterpret_cast<char*>(row.data()), dim * sizeof(double));
        }

        // Вивід прогресу кожні 1000 рядків
        if (a_val % 1000 == 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

            std::cout << "\rRound 2 progress: processed " << a_val
                << " rows out of " << dim
                << ", elapsed time: " << elapsed << " s" << std::flush;
        }
    }

    out_r.close();

    std::cout << "\rRound 2 completed. Total rows processed: " << dim << "                     \n";

    log << "rho_perm_LBlock, p_perm_LBlock, 2,"
        << std::setprecision(16) << max_t2 << ","
        << a_max2 << "," << b_max2 << ","
        << Template(a_max2) << "," << Template(b_max2) << "\n";
    std::cout << "Round 2: max_val = " << max_t2 << ", a_max = " << a_max2 << ", b_max = " << b_max2 << "\n";

    // --- Round 3 and beyond ---
    for (int t = 3; t <= rounds; ++t) {
        start_time = std::chrono::steady_clock::now();
        std::string file_prev = round_file + std::to_string(t - 1) + ".bin";
        std::string file_curr = round_file + std::to_string(t) + ".bin";

        std::ifstream in_prev(file_prev, std::ios::binary);
        if (!in_prev.is_open()) {
            std::cerr << "Cannot open previous round file: " << file_prev << "\n";
            return;
        }

        std::ofstream out_curr(file_curr, std::ios::binary);
        if (!out_curr.is_open()) {
            std::cerr << "Cannot create file for round " << t << ": " << file_curr << "\n";
            return;
        }

        double max_val = 0.0;
        TemplateID a_max = 0, b_max = 0;

#pragma omp parallel for schedule(dynamic)
        for (TemplateID a_val = 0; a_val < dim; ++a_val) {
            std::vector<double> row(dim, 0.0);
            std::vector<double> UB_prev_row(dim);

#pragma omp critical
            {
                in_prev.seekg(static_cast<std::streamoff>(a_val) * dim * sizeof(double));
                in_prev.read(reinterpret_cast<char*>(UB_prev_row.data()), dim * sizeof(double));
            }

            for (TemplateID b_val = 0; b_val < dim; ++b_val) {
                Template b(b_val);
                miniTemplate b_x, b_y;
                for (int i = 0; i < num_blocks / 2; ++i) b_x[i] = b[i];
                for (int i = 0; i < num_blocks / 2; ++i) b_y[i] = b[i + num_blocks / 2];

                if (b_x.none()) {
                    miniTemplate rho_inv_by = RhoInverse(b_y);
                    Template transformed_b;
                    for (int i = 0; i < num_blocks / 2; ++i)
                        transformed_b[i] = rho_inv_by[i];

                    row[b_val] = UB_prev_row[transformed_b.to_ulong()];
                }
                else {
                    double M = 0.0;
                    for (TemplateID gamma_val = 0; gamma_val < dim; ++gamma_val) {
                        std::vector<double> gamma_row(dim);
#pragma omp critical
                        {
                            in_prev.seekg(static_cast<std::streamoff>(gamma_val) * dim * sizeof(double));
                            in_prev.read(reinterpret_cast<char*>(gamma_row.data()), dim * sizeof(double));
                        }
                        M = std::max(M, gamma_row[b_val]);
                    }

                    miniTemplate Pb_x = P(b_x);
                    TemplateID phi_index = (b_y.to_ulong() << (num_blocks / 2)) + Pb_x.to_ulong();
                    double sum = 0.0;

                    if (phi_index < Phi_system.size()) {
                        const auto& phi_set = Phi_system[phi_index];

                        for (TemplateID gamma_val_raw = 0; gamma_val_raw < (1 << (num_blocks / 2)); ++gamma_val_raw) {
                            miniTemplate gamma;
                            for (int i = 0; i < num_blocks / 2; ++i)
                                gamma[i] = (gamma_val_raw >> i) & 1;

                            miniTemplate rho_gamma = Rho(gamma);
                            if (phi_set.find(rho_gamma) != phi_set.end()) {
                                Template gamma_template;
                                for (int i = 0; i < num_blocks / 2; ++i)
                                    gamma_template[i] = gamma[i];
                                for (int i = 0; i < num_blocks / 2; ++i)
                                    gamma_template[i + num_blocks / 2] = b_x[i];

                                TemplateID gamma_id = gamma_template.to_ulong();
                                int gamma_wt = static_cast<int>(gamma.count());
                                int bx_wt = hammingWeight(b_x);
                                size_t limit = static_cast<size_t>(std::pow((1 << word_size) - 1, gamma_wt));
                                double sum_u = get_precomputed_sum(bx_wt, limit, precomputed_sums);

                                sum += UB_prev_row[gamma_id] * sum_u;
                            }
                        }
                    }

                    row[b_val] = std::min(M, sum);
                }

#pragma omp critical
                if (a_val != 0 && row[b_val] > max_val) {
                    max_val = row[b_val];
                    a_max = a_val;
                    b_max = b_val;
                }
            }

#pragma omp critical
            {
                out_curr.seekp(static_cast<std::streamoff>(a_val) * dim * sizeof(double));
                out_curr.write(reinterpret_cast<char*>(row.data()), dim * sizeof(double));
            }

            // Вивід прогресу кожні 1000 рядків у циклі раунду t
            if (a_val % 1000 == 0) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

                std::cout << "\rRound " << t << " progress: processed " << a_val
                    << " rows out of " << dim
                    << ", elapsed time: " << elapsed << " s" << std::flush;
            }
        } 

        std::cout << "\rRound " << t << " completed. Total rows processed: " << dim << "                     \n";

        in_prev.close();
        std::remove(file_prev.c_str());
        out_curr.close();

        log << "rho_perm_LBlock, p_perm_LBlock, " << t << ","
            << std::setprecision(16) << max_val << ","
            << a_max << "," << b_max << ","
            << Template(a_max) << "," << Template(b_max) << "\n";
        std::cout << "Round " << t << ": max_val = " << max_val << ", a_max = " << a_max << ", b_max = " << b_max << "\n";
    }

    std::cout << "UB computation finished and saved round by round to .bin files.\n";
}


//void computeUB_to_bin(
//    const std::vector<std::vector<miniTemplate>>& Phi_system,
//    int rounds,
//    const std::string& output_prefix,
//    double p,
//    std::ofstream& log,
//    const std::vector<std::map<double, size_t>>& u_distributions,
//    const std::vector<std::vector<double>>& precomputed_sums
//) {
//    constexpr size_t N = 1 << num_blocks;
//    std::string prev_filename = output_prefix + "_round_2.bin";
//
//    // === Step 1: Compute UB^{[2]} ===
//    {
//        std::ofstream bin_out(prev_filename, std::ios::binary);
//        double max_val = 0.0;
//        TemplateID max_a = 0, max_b = 0;
//
//        for (TemplateID a_val = 0; a_val < N; ++a_val) {
//            Template a(a_val);
//            miniTemplate a_x, a_y;
//            for (int i = 0; i < num_blocks / 2; ++i) {
//                a_x[i] = a[i];
//                a_y[i] = a[i + num_blocks / 2];
//            }
//
//            miniTemplate Pa_y = P(a_y);
//            miniTemplate rho_ax = Rho(a_x);
//            miniTemplate rho_ay = Rho(a_y);
//
//            for (TemplateID b_val = 0; b_val < N; ++b_val) {
//                if (a_val % 100 == 0 && b_val == 0) {
//                    std::cout << "[Round 2] UB row a_val = " << a_val << " / " << N << "\n";
//                }
//                Template b(b_val);
//                miniTemplate b_x, b_y;
//                for (int i = 0; i < num_blocks / 2; ++i) {
//                    b_x[i] = b[i];
//                    b_y[i] = b[i + num_blocks / 2];
//                }
//
//                TemplateID line1 = (Pa_y.to_ullong() << 8) + rho_ax.to_ullong();
//                TemplateID line2 = (b_x.to_ullong() << 8) + rho_ay.to_ullong();
//
//                bool cond1 = line1 < Phi_system.size() &&
//                    std::find(Phi_system[line1].begin(), Phi_system[line1].end(), b_x) != Phi_system[line1].end();
//                bool cond2 = line2 < Phi_system.size() &&
//                    std::find(Phi_system[line2].begin(), Phi_system[line2].end(), b_y) != Phi_system[line2].end();
//
//                double ub = (cond1 && cond2) ? std::pow(p, hammingWeight(a_y) + hammingWeight(b_x)) : 0.0;
//                bin_out.write(reinterpret_cast<const char*>(&ub), sizeof(double));
//
//                if (!(a_val == 0 && b_val == 0) && ub > max_val) {
//                    max_val = ub;
//                    max_a = a_val;
//                    max_b = b_val;
//                }
//            }
//        }
//
//        log << "rho_perm_LBlock,p_perm_LBlock,2,"
//            << std::setprecision(16) << max_val << ","
//            << max_a << "," << max_b << ","
//            << Template(max_a) << "," << Template(max_b) << "\n";
//    }
//
//    // === Step 2: UB^t for t >= 3 ===
//    for (int t = 3; t <= rounds; ++t) {
//        std::string current_filename = output_prefix + "_round_" + std::to_string(t) + ".bin";
//        std::ifstream bin_in(prev_filename, std::ios::binary);
//        std::ofstream bin_out(current_filename, std::ios::binary);
//        if (!bin_in || !bin_out) {
//            std::cerr << "Cannot open UB bin file for round " << t << "\n";
//            return;
//        }
//
//        double max_val = 0.0;
//        TemplateID max_a = 0, max_b = 0;
//
//        std::vector<double> row_cache(N);     // for fixed a_val
//        std::vector<double> column_cache(N);  // for fixed b_val
//
//        for (TemplateID b_val = 0; b_val < N; ++b_val) {
//            for (TemplateID a_val = 0; a_val < N; ++a_val) {
//                bin_in.seekg((a_val * N + b_val) * sizeof(double), std::ios::beg);
//                bin_in.read(reinterpret_cast<char*>(&column_cache[a_val]), sizeof(double));
//            }
//
//            for (TemplateID a_val = 0; a_val < N; ++a_val) {
//                if (a_val % 100 == 0) {
//                    std::cout << "[Round " << t << "] UB row a_val = " << a_val << " / " << N << "\n";
//                }
//
//                bin_in.seekg(a_val * N * sizeof(double), std::ios::beg);
//                bin_in.read(reinterpret_cast<char*>(row_cache.data()), N * sizeof(double));
//
//                Template b(b_val);
//                miniTemplate b_x, b_y;
//                for (int i = 0; i < num_blocks / 2; ++i) {
//                    b_x[i] = b[i];
//                    b_y[i] = b[i + num_blocks / 2];
//                }
//
//                double result = 0.0;
//                if (b_x.none()) {
//                    miniTemplate rho_inv_by = RhoInverse(b_y);
//                    Template transformed_b;
//                    for (int i = 0; i < num_blocks / 2; ++i) {
//                        transformed_b[i] = rho_inv_by[i];
//                        transformed_b[i + num_blocks / 2] = 0;
//                    }
//                    TemplateID transformed_b_val = transformed_b.to_ulong();
//                    result = row_cache[transformed_b_val];
//                }
//                else {
//                    miniTemplate Pb_x = P(b_x);
//                    TemplateID phi_index = (b_y.to_ulong() << 8) + Pb_x.to_ullong();
//                    int bx_weight = hammingWeight(b_x);
//                    double sum = 0.0;
//
//                    if (phi_index < Phi_system.size()) {
//                        const auto& phi_set = Phi_system[phi_index];
//                        for (TemplateID gamma_val = 0; gamma_val < (1 << (num_blocks / 2)); ++gamma_val) {
//                            miniTemplate gamma(gamma_val);
//                            miniTemplate rho_gamma = Rho(gamma);
//                            if (std::find(phi_set.begin(), phi_set.end(), rho_gamma) != phi_set.end()) {
//                                Template gamma_template;
//                                for (int i = 0; i < num_blocks / 2; ++i) gamma_template[i] = gamma[i];
//                                for (int i = 0; i < num_blocks / 2; ++i) gamma_template[i + num_blocks / 2] = b_x[i];
//                                TemplateID gamma_template_val = gamma_template.to_ulong();
//
//                                double prev_val = row_cache[gamma_template_val];
//
//                                int gamma_weight = hammingWeight(gamma);
//                                size_t limit = static_cast<size_t>(std::pow((1 << word_size) - 1, gamma_weight));
//                                double sum_u = get_precomputed_sum(bx_weight, limit, precomputed_sums);
//                                sum += prev_val * sum_u;
//                            }
//                        }
//                    }
//
//                    double M = *std::max_element(column_cache.begin(), column_cache.end());
//                    result = std::min(M, sum);
//                }
//
//                bin_out.write(reinterpret_cast<const char*>(&result), sizeof(double));
//
//                if (!(a_val == 0 && b_val == 0) && result > max_val) {
//                    max_val = result;
//                    max_a = a_val;
//                    max_b = b_val;
//                }
//            }
//        }
//
//        bin_in.close();
//        bin_out.close();
//        std::remove(prev_filename.c_str());
//        prev_filename = current_filename;
//
//        log << "rho_perm_LBlock,p_perm_LBlock," << t << ","
//            << std::setprecision(16) << max_val << ","
//            << max_a << "," << max_b << ","
//            << Template(max_a) << "," << Template(max_b) << "\n";
//    }
//
//    std::cout << "UB computation completed and saved to .bin files\n";
//}
