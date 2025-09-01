#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "resources.hpp"

namespace fs = std::filesystem;

// Helper function to construct paths relative to project root
std::string get_project_path(const std::string& relative_path) {
    return std::filesystem::path(PROJECT_ROOT) / relative_path;
}

class GmpGpuTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {
        // Clean up test output file
        if (fs::exists(get_project_path("test/test_files/gmpFeatures_test.dat"))) {
            fs::remove(get_project_path("test/test_files/gmpFeatures_test.dat"));
        }
        if (fs::exists(get_project_path("test/test_files/test_config.json"))) {
            fs::remove(get_project_path("test/test_files/test_config.json"));
        }
    }

    std::vector<double> readFileAsDoubles(const std::string& filename) {
        std::ifstream file(filename);
        std::vector<double> values;
        
        if (!file.is_open()) {
            return values;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            
            // Split by comma and read each value
            while (std::getline(ss, token, ',')) {
                // Remove leading/trailing whitespace
                token.erase(0, token.find_first_not_of(" \t\r\n"));
                token.erase(token.find_last_not_of(" \t\r\n") + 1);
                
                if (!token.empty()) {
                    try {
                        double value = std::stod(token);
                        values.push_back(value);
                    } catch (const std::exception& e) {
                        // Skip invalid values
                        continue;
                    }
                }
            }
        }
        
        return values;
    }

    double computeNorm(const std::vector<double>& vec) {
        double sum = 0.0;
        for (double val : vec) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    double computeNormDifference(const std::vector<double>& vec1, const std::vector<double>& vec2) {
        if (vec1.size() != vec2.size()) {
            return std::numeric_limits<double>::infinity();
        }
        
        double sum = 0.0;
        for (size_t i = 0; i < vec1.size(); ++i) {
            double diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    bool fileExists(const std::string& filename) {
        return fs::exists(filename);
    }
};

// Test: Run the application with example files and compare output
TEST_F(GmpGpuTest, CompareWithExpectedOutput) {
    // Create a test config that points to example files but outputs to test directory
    std::ofstream test_config(get_project_path("test/test_files/test_config.json"));
    test_config << R"({
        "system file path": ")" << get_project_path("test/test_files/test.cif") << R"(",
        "psp file path": ")" << get_project_path("test/test_files/QE-kjpaw.gpsp") << R"(",
        "output file path": ")" << get_project_path("test/test_files/gmpFeatures_test.dat") << R"(",
        "square": 0,
        "cutoff": 0.0,
        "overlap threshold": 1e-11,
        "cutoff method": 4,
        "scaling mode": 0,
        "orders": [-1, 0, 1, 2],
        "sigmas": [0.1, 0.2, 0.3]
    })";
    test_config.close();

    // Run the application with absolute paths
    std::string command = get_project_path("build/gmp-featurizer") + " " + get_project_path("test/test_files/test_config.json");
    int result = std::system(command.c_str());
    
    // Check that the application ran successfully
    EXPECT_EQ(result, 0) << "Application failed to run successfully";
    
    // Check that output file was created
    EXPECT_TRUE(fileExists(get_project_path("test/test_files/gmpFeatures_test.dat"))) 
        << "Output file was not created";
    
    // Read the expected output based on precision
#ifdef GMP_USE_SINGLE_PRECISION
    std::vector<double> expected_output = readFileAsDoubles(get_project_path("test/test_files/gmpFeatures_float.dat"));
#else
    std::vector<double> expected_output = readFileAsDoubles(get_project_path("test/test_files/gmpFeatures_double.dat"));
#endif
    EXPECT_FALSE(expected_output.empty()) << "Expected output file is empty or not found";
    
    // Read the actual output
    std::vector<double> actual_output = readFileAsDoubles(get_project_path("test/test_files/gmpFeatures_test.dat"));
    EXPECT_FALSE(actual_output.empty()) << "Actual output file is empty";
    
    // Check that both arrays have the same size
    EXPECT_EQ(actual_output.size(), expected_output.size()) 
        << "Output arrays have different sizes";
    
    // Compute the norm of the difference
    double norm_diff = computeNormDifference(actual_output, expected_output);
    
    // Check that the difference is within tolerance (adjust tolerance based on precision)
#ifdef GMP_USE_SINGLE_PRECISION
    double tolerance = 1e-5;  // Larger tolerance for single precision
#else
    double tolerance = 1e-10; // Smaller tolerance for double precision
#endif
    EXPECT_LT(norm_diff, tolerance) 
        << "Norm of difference (" << norm_diff << ") exceeds tolerance (" << tolerance << ")";
    
    std::cout << "Norm of difference: " << norm_diff << std::endl;
    std::cout << "Expected output size: " << expected_output.size() << std::endl;
    std::cout << "Actual output size: " << actual_output.size() << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef GMP_ENABLE_CUDA
    // Explicitly cleanup CUDA resources before exit
    gmp::resources::gmp_resource::instance().cleanup();
#endif
    return result;
} 