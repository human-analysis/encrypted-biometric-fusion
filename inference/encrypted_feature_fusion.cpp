/*
 Author: Luke Sperling
 Created: 10-01-21
 Modified: 07-08-22
 Inference-time linear feature-level fusion of multimodal templates.
 Feature vectors are concatenated together, linearly projected to a new dimensionality, and normalized, entirely in the encrypted domain.
 Privacy is preserved at every stage of this process
 */
#include <cmath>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include "seal/seal.h"
//#include "utils.h"

using namespace std;
using namespace std::chrono;
using namespace seal;

bool zero_vector(vector<double> input)
{
    for(int i = 0; i < input.size(); i++)
    {
        if(abs(input[i]) > 0.0001)
            return false;
    }
    return true;
}


void encrypted_feature_fusion_polynomial_approximation_arbitrary(string P_file_name_in, string normalized_outfile_name, int degree)
{

    //Encryption parameters chosen based on multiplicative depth needed
    
    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    //size_t poly_modulus_degree = 16384;
    //poly_modulus_degree = 32768;//
    size_t poly_modulus_degree = 32768;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    
    if(degree==2)
    {
        poly_modulus_degree = 16384;//
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 40, 40, 40, 40, 40, 40, 40, 40, 50 }));
    }
    else if(degree==6)
    {
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50 }));
    }
    
    
    double scale = pow(2.0, 40);

    SEALContext context(parms);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    //print_parameters(context);
    cout << endl;
    cout << "time to create context: " << duration.count() / 1000.0 << " milliseconds" << endl;

    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    
    slot_count = 1024;
    cout << "Number of slots: " << slot_count << endl;
    
    //set up coefficients for normalization
    size_t slots = encoder.slot_count();
    vector<double> a1, b1, c1, d1, a2, b2, c2, d2;
    Plaintext a1_plain, b1_plain, c1_plain, d1_plain, a2_plain, b2_plain, c2_plain, d2_plain;
    
    for(int i=0; i<slots; i++)
    {
        
        if(degree==1)
        {
            a1.push_back(2.78221164);
            b1.push_back(-2.61776258);
        }
        else if(degree==2)
        {
            a1.push_back(2.51308415);
            b1.push_back(-1.81897596);
            c1.push_back(0.42084296);
        }
        else if (degree==3)
        {
            a1.push_back(4.38423127);
            b1.push_back(-13.57853979);
            c1.push_back(19.8459398);
            d1.push_back(-9.81663423);
        }
        else if (degree==6)
        {
            a1.push_back(0.33009964);
            b1.push_back(3.75046592);
            c1.push_back(-2.53130775);
            d1.push_back(0.60632975);
            
            a2.push_back(5.23381489);
            b2.push_back(-3.742239);
            c2.push_back(1.00104718);
            d2.push_back(-0.08817609);
        }
        
    }
    
    encoder.encode(a1, scale, a1_plain);
    encoder.encode(b1, scale, b1_plain);
    encoder.encode(c1, scale, c1_plain);
    encoder.encode(d1, scale, d1_plain);
    encoder.encode(a2, scale, a2_plain);
    encoder.encode(b2, scale, b2_plain);
    encoder.encode(c2, scale, c2_plain);
    encoder.encode(d2, scale, d2_plain);
    
    vector<Plaintext> coeffs;
    coeffs.push_back(a1_plain);
    coeffs.push_back(b1_plain);
    if(degree>1)
        coeffs.push_back(c1_plain);
    if(degree>2)
        coeffs.push_back(d1_plain);
    if(degree>=6)
    {
        coeffs.push_back(a2_plain);
        coeffs.push_back(b2_plain);
        coeffs.push_back(c2_plain);
        coeffs.push_back(d2_plain);
    }
    
    vector<vector<Plaintext>> coeff_list;
    if(degree==2)
    {
        vector<Plaintext> t;
        t.push_back(c1_plain);
        t.push_back(b1_plain);
        t.push_back(a1_plain);
        coeff_list.push_back(t);
    }
    if(degree==6)
    {
        vector<Plaintext> t1;
        t1.push_back(d1_plain);
        t1.push_back(c1_plain);
        t1.push_back(b1_plain);
        t1.push_back(a1_plain);
        coeff_list.push_back(t1);
        
        vector<Plaintext> t2;
        t2.push_back(d2_plain);
        t2.push_back(c2_plain);
        t2.push_back(b2_plain);
        t2.push_back(a2_plain);
        coeff_list.push_back(t2);
    }
    
    //this value MUST be manually changed depending on the degree used for normalization
    
    vector<int> degrees{3};
    //vector<int> degrees{3,3};

    
    
    //read data file to get A values
    vector<vector<double>> A_message;
    vector<Plaintext> A_plain;
    vector<Ciphertext> A_enc;
    A_enc.reserve(20);
    
    string row;
    ifstream A_file ("../../train/data/dataset/A_values_test.txt");
    string A_values_str;
    int zeroes_to_add;
    int to_rotate = -1; //we need to know how much to rotate B by for concatenation
    
    
    //we can pack many vectors into a single ciphertext
    //each vector takes its size (slot_count) twice to allow for rotation to work
    int max_packed_words = slots / (2 * slot_count);
    //this next value captures the case where we have fewer samples than max_packed_words
    int max_words_packed_in_single_cipher = -1;
    int packed_words = 0;
    
    cout << "we can fit this many words in a cipher:" << max_packed_words << endl;
    
    cout << "build A" << endl;
    vector<double> message_temp;
    message_temp.reserve(16384);
    Plaintext plain_temp;
    Ciphertext enc_temp;
    
    if (A_file.is_open())
    {
      while ( getline (A_file,row) )
      {
          //remove commas, source: https://stackoverflow.com/questions/20326356/how-to-remove-all-the-occurrences-of-a-char-in-c-string
          row.erase(remove(row.begin(), row.end(), ','), row.end());
          
          zeroes_to_add = slot_count;
          //strip off start and end brackets
          row = row.substr(1,A_values_str.length()-2);
          size_t pos = 0;
          string A_sub_value;
          while((pos = row.find(' ')) != -1)
          {
              zeroes_to_add--;
              A_sub_value = row.substr(0, pos-1);
              double A_sub_value_final = stod(A_sub_value);
              message_temp.push_back(A_sub_value_final);
              row.erase(0, pos+1);
          }
          for(int i = 0; i < zeroes_to_add; i++)
              message_temp.push_back(0.0);
          if(to_rotate == -1)
          {
              to_rotate = -1 * (slot_count - zeroes_to_add);
          }
          
          int to_replicate = slot_count;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          packed_words++;
          if(packed_words >= max_packed_words)
          {
              if(packed_words>max_words_packed_in_single_cipher)
                  max_words_packed_in_single_cipher = packed_words;
              encoder.encode(message_temp, scale, plain_temp);

              encryptor.encrypt(plain_temp, enc_temp);
              A_enc.push_back(enc_temp);
              A_message.push_back(message_temp);
              message_temp.clear();
              packed_words = 0;
          }
      }
    if(message_temp.size() > 0)
    {
        if(packed_words>max_words_packed_in_single_cipher)
            max_words_packed_in_single_cipher = packed_words;
        encoder.encode(message_temp, scale, plain_temp);
        encryptor.encrypt(plain_temp, enc_temp);
        A_enc.push_back(enc_temp);
        A_plain.push_back(plain_temp);
        A_message.push_back(message_temp);
        message_temp.clear();
        packed_words = 0;
    }
    A_file.close();
    }

    //read data file to get B values
    vector<vector<double>> B_message;
    vector<Plaintext> B_plain;
    vector<Ciphertext> B_enc;
    B_enc.reserve(20);
    ifstream B_file ("../../train/data/dataset/B_values_test.txt");
    string B_values_str;
    
    cout << "build B" << endl;
    int test_counter = 0;
    if (B_file.is_open())
    {
      while ( getline (B_file,row) )
      {
          test_counter ++;
          //remove commas, source: https://stackoverflow.com/questions/20326356/how-to-remove-all-the-occurrences-of-a-char-in-c-string
          row.erase(remove(row.begin(), row.end(), ','), row.end());
          zeroes_to_add = slot_count;
          row = row.substr(1,B_values_str.length()-2);
          size_t pos = 0;
          string B_sub_value;
          while((pos = row.find(' ')) != -1)
          {
              zeroes_to_add--;
              B_sub_value = row.substr(0, pos-1);
              double B_sub_value_final = stod(B_sub_value);
              message_temp.push_back(B_sub_value_final);
              row.erase(0, pos+1);
          }
          
          for(int i = 0; i < zeroes_to_add; i++)
              message_temp.push_back(0.0);
          int to_replicate = slot_count;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          
          packed_words++;
          if(packed_words >= max_packed_words)
          {
              encoder.encode(message_temp, scale, plain_temp);
              encryptor.encrypt(plain_temp, enc_temp);
              B_enc.push_back(enc_temp);
              B_message.push_back(message_temp);
              message_temp.clear();
              packed_words = 0;
          }
          
      }
    if(message_temp.size() > 0)
    {
        encoder.encode(message_temp, scale, plain_temp);
        encryptor.encrypt(plain_temp, enc_temp);
        B_enc.push_back(enc_temp);
        B_plain.push_back(plain_temp);
        B_message.push_back(message_temp);
        message_temp.clear();
        packed_words = 0;
    }
        
        
    B_file.close();
    }
    
    cout << "B size: " << B_message.size() << endl;
    cout << test_counter << endl;
    
    cout << "concatenate dataset" << endl;
    start = high_resolution_clock::now();
    
    //build encrypted dataset by rotating each query in B, adding together
    //this is concatenation of each row of A and B
    cout << to_rotate << endl;
    vector<Ciphertext> enc_queries;
    enc_queries.reserve(20);
    for(int i = 0; i < A_enc.size(); i++)
    {
        Ciphertext query;
        Ciphertext B_temp = B_enc[i];
        evaluator.rotate_vector(B_enc[i], to_rotate, gal_keys, B_temp);
        evaluator.add(A_enc[i], B_temp, query);
        enc_queries.push_back(query);
    }
    
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to concatenate: " << duration.count() / 1000.0 << " milliseconds" << endl;
    //concatenation done
    

    vector<Ciphertext> p_matrix_diagonal;
    vector<Plaintext> p_matrix_diagonal_plain;
    vector<vector<double>> p_matrix_diagonal_message;
    p_matrix_diagonal.reserve(slot_count);
    
    cout << "building P" << endl;
    ifstream P_file(P_file_name_in);
    
    string P_values_str;
    message_temp.clear();
    packed_words = 0;
    
    int gamma = 0;
    
    if (P_file.is_open())
    {
      while ( getline (P_file,row) )
      {
          gamma++;
          size_t pos = 0;
          string P_sub_value;
          while((pos = row.find(' ')) != -1)
          {
              P_sub_value = row.substr(0, pos-1);
              double P_sub_value_final = stod(P_sub_value);
              message_temp.push_back(P_sub_value_final);
              row.erase(0, pos+1);
          }

          int to_replicate = message_temp.size();
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[i]);
          
          for(int i=0;i<(max_packed_words-1);i++)
          {
              for(int j=0;j<to_replicate*2;j++)
                  message_temp.push_back(message_temp[j]);
          }
          encoder.encode(message_temp, scale, plain_temp);
          p_matrix_diagonal_plain.push_back(plain_temp);
          p_matrix_diagonal_message.push_back(message_temp);
          
          message_temp.clear();
          
      }

    P_file.close();
    }
    
    cout << "Gamma = " << gamma << endl;

    //p matrix is d_0 by d_1
    int d_0 = gamma;
    int d_1 = 1024;
    int logd_0 = log2(d_0);
    int logd_1 = log2(d_1);
    
    cout << "begin fusion" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    start = high_resolution_clock::now();
    
    vector<Ciphertext> enc_fusions;
    enc_fusions.reserve(20);
    vector<double> message_zeroes;
    vector<double> message_ones;
    vector<double> message_zeroes2;
    Plaintext plain_zeroes;
    Plaintext plain_ones;
    Plaintext plain_zeroes2;
    
    message_zeroes.reserve(slots);
    message_ones.reserve(slots);
    message_zeroes2.reserve(slots);

    for (int i = 0; i < slots; i++)
    {
        message_zeroes.push_back(0.0);
        message_ones.push_back(1.0);
        message_zeroes2.push_back(0.0);
    }
    encoder.encode(message_zeroes, scale, plain_zeroes);
    encoder.encode(message_ones, scale, plain_ones);
    encoder.encode(message_zeroes2, scale, plain_zeroes2);
    
    for(int c = 0; c < enc_queries.size(); c++)
    {
        cout << c << " ";
        Ciphertext encrypted_result;
        encryptor.encrypt(plain_zeroes, encrypted_result);
        
        evaluator.multiply_plain(encrypted_result, plain_ones, encrypted_result);
        evaluator.relinearize_inplace(encrypted_result, relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result); //level of encrypted_result must match sub result, which will be rescaled once
        
        Ciphertext encrypted_sub_result;
        
        for(int i = 0; i < d_0; i++)
        {
            if(!zero_vector(p_matrix_diagonal_message[i])) //cannot plainmult a vector of all zeroes
            {

                evaluator.multiply_plain(enc_queries[c], p_matrix_diagonal_plain[i], encrypted_sub_result);
                evaluator.relinearize_inplace(encrypted_sub_result, relin_keys);
                evaluator.rescale_to_next_inplace(encrypted_sub_result);
                evaluator.add_inplace(encrypted_result, encrypted_sub_result);
            }
            else
            {
                cout << "zero vector detected" << endl;
            }
            evaluator.rotate_vector(enc_queries[c], 1, gal_keys, enc_queries[c]);
        }
        Ciphertext encrypted_final_result;
        encryptor.encrypt(plain_zeroes2, encrypted_final_result);

        evaluator.multiply_plain(encrypted_final_result, plain_ones, encrypted_final_result);
        evaluator.relinearize_inplace(encrypted_final_result, relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_final_result); //level of encrypted_final_result must match result, which will be rescaled once
        evaluator.add_inplace(encrypted_final_result, encrypted_result);
        for(int j = logd_1-1; j>=logd_0;j--)
        {
            evaluator.rotate_vector(encrypted_result, pow(2,j), gal_keys, encrypted_result);
            Ciphertext temp;
            evaluator.add(encrypted_final_result, encrypted_result, temp);
            encrypted_result = temp;
            encrypted_final_result = temp;
        }
        
        enc_fusions.push_back(encrypted_final_result);
    }
    
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to fuse: " << duration.count() / 1000.0 << " milliseconds" << endl;
    //Linear projection done
    
    
    //Normalization begins here
    cout << "begin normalization" << endl;
    
    
    vector<Ciphertext> normalized_enc_fusions;
    
    start = high_resolution_clock::now();
    for(int i=0; i < enc_fusions.size(); i++)
    {
        Ciphertext cipher = enc_fusions[i];
        
        int vector_size = gamma;
        //Homormorphic inner product to find squared norm of input
        int iterations = log2(vector_size);
        Ciphertext squared_norm;
        evaluator.square(cipher, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        
        for(int i = 0; i < iterations; i++)
        {
            Ciphertext temp;
            evaluator.rotate_vector(squared_norm, pow(2, i), gal_keys, temp);
            evaluator.add(squared_norm, temp, squared_norm);
        }
        
        //end HE inner product
        
        //begin arbitrary polynomial evaluation
        Ciphertext result;
        
        Ciphertext x = squared_norm;

        for(int c = 0; c < coeff_list.size(); c++)
        {
            //encryptor.encrypt(coeff_list[c][0], result);
            evaluator.mod_switch_to_inplace(coeff_list[c][0], x.parms_id());
            coeff_list[c][0].scale() = scale;
            x.scale() = scale;
            evaluator.multiply_plain(x, coeff_list[c][0], result);
            evaluator.relinearize_inplace(result, relin_keys);
            evaluator.rescale_to_next_inplace(result);
            
            evaluator.mod_switch_to_inplace(coeff_list[c][1], result.parms_id());
            coeff_list[c][1].scale() = scale;
            result.scale() = scale;
            evaluator.add_plain(result, coeff_list[c][1], result);
            
            
            for(int d = 2; d < coeff_list[c].size(); d++)
            {
                evaluator.mod_switch_to_inplace(x, result.parms_id());
                x.scale() = scale;
                result.scale() = scale;
                evaluator.multiply(x, result, result);
                evaluator.relinearize_inplace(result, relin_keys);
                evaluator.rescale_to_next_inplace(result);
                
                evaluator.mod_switch_to_inplace(coeff_list[c][d], result.parms_id());
                coeff_list[c][d].scale() = scale;
                result.scale() = scale;
                evaluator.add_plain(result, coeff_list[c][d], result);
            }
            x = result;
        }
        //end arbitrary polynomial evaluation
        
        //now we multiply the original vector by the inverse norm and return that result
        Ciphertext normalized_cipher;
        
        evaluator.mod_switch_to_inplace(cipher, x.parms_id());
        cipher.scale() = x.scale();
        
        evaluator.multiply(cipher, x, normalized_cipher);
        evaluator.relinearize_inplace(normalized_cipher, relin_keys);
        evaluator.rescale_to_next_inplace(normalized_cipher);
        normalized_enc_fusions.push_back(normalized_cipher);
    }
    
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to normalize: " << duration.count() / 1000.0 << " milliseconds" << endl;
    cout << "normalization complete" << endl;
    //end normalization
    
    int delta = 1024;
    
    //preprocessing step
    start = high_resolution_clock::now();
    vector<double> mask_message;
    Plaintext mask_plain;
    
    for(int i = 0; i<max_words_packed_in_single_cipher; i++)
    {
        for(int j=0; j<gamma; j++)
            mask_message.push_back(1);
        for(int j=0; j<delta-gamma; j++)
            mask_message.push_back(0);
    }
    
    encoder.encode(mask_message, scale, mask_plain);
    
    evaluator.mod_switch_to_inplace(mask_plain, normalized_enc_fusions[0].parms_id());
    mask_plain.scale() = normalized_enc_fusions[0].scale();
    
    for(int c = 0; c < normalized_enc_fusions.size(); c++)
    {
        evaluator.multiply_plain_inplace(normalized_enc_fusions[c], mask_plain);
        evaluator.relinearize_inplace(normalized_enc_fusions[c], relin_keys);
        evaluator.rescale_to_next_inplace(normalized_enc_fusions[c]);
    }
    
    //gallery creation
    vector<Ciphertext> gallery;
    cout << "delta:" << delta << " gamma:" << gamma << endl;
    int packing_number = delta/gamma;
    int curr = 0;
    Ciphertext gallery_template;
    for(int c = 0; c < normalized_enc_fusions.size(); c++)
    {
        if(curr==0)
        {
            gallery_template = normalized_enc_fusions[c];
        }
        else
        {
            Ciphertext temp;
            evaluator.rotate_vector(normalized_enc_fusions[c], gamma * c, gal_keys, temp);
            evaluator.add(gallery_template, temp, gallery_template);
        }
        curr++;
        if(curr>=gamma)
        {
            curr = 0;
            gallery.push_back(gallery_template);
        }
    }
    if(curr > 0)
    {
        gallery.push_back(gallery_template);
    }
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to preprocess: " << duration.count() / 1000.0 << " milliseconds" << endl;
    //end preprocess
    
    
    //begin match score calculation
    Ciphertext probe = normalized_enc_fusions[0];
    
    start = high_resolution_clock::now();
    for(int c = 0; c < normalized_enc_fusions.size(); c++)
    {
        Ciphertext enc_score_result;
        evaluator.multiply(normalized_enc_fusions[c], probe, enc_score_result);
        evaluator.relinearize_inplace(enc_score_result, relin_keys);
        evaluator.rescale_to_next_inplace(enc_score_result);
        
        for(int i = 0; i < log2(gamma); i++)
        {
            Ciphertext temp;
            evaluator.rotate_vector(enc_score_result, pow(2, i), gal_keys, temp);
            evaluator.add_inplace(enc_score_result, temp);
        }
    }
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to match score: " << duration.count() / 1000.0 << " milliseconds" << endl;
    //end match score calculation
    
    
    
    //output results to a file
    ofstream myfile;
    myfile.open (normalized_outfile_name);
    for(int c = 0; c < normalized_enc_fusions.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(normalized_enc_fusions[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            for(int j = slot_count*2*i; j < slot_count*2*i+gamma; j++)
            {
                myfile << result[j] << " ";
            }
            myfile << endl;
        }
    }
    myfile.close();

}




int main()
{
    encrypted_feature_fusion_polynomial_approximation_arbitrary("../../train/data/degree=2strict/diagonal/XXX.txt","../results/normalized_encrypted_results_lambda=XXX_margin=XXX_gamma=XXX.txt",2);
    return 0;
}
