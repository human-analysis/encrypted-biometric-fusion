// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "examples.h"
#include <cmath>
#include <chrono>
#include <algorithm>

#include <filesystem>
#include <unistd.h>

#include <string>

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

Ciphertext hybrid_matmul(vector<Plaintext> matrix, vector<vector<double>> matrix_message, Ciphertext query_enc, int d_0, int d_1, Encryptor* encryptor, Evaluator* evaluator, RelinKeys relin_keys, GaloisKeys gal_keys, Plaintext plain_zeroes, Plaintext plain_ones)
{
    int logd_0 = log2(d_0);
    int logd_1 = log2(d_1);
    Ciphertext encrypted_result;
    //HERE
    encryptor->encrypt(plain_zeroes, encrypted_result);
    
    evaluator->multiply_plain(encrypted_result, plain_ones, encrypted_result);
    evaluator->relinearize_inplace(encrypted_result, relin_keys);
    evaluator->rescale_to_next_inplace(encrypted_result); //level of encrypted_result must match sub result, which will be rescaled once

    Ciphertext encrypted_sub_result;

    for(int i = 0; i < d_0; i++)
    {
        //cout << "iteration " << i << " of mat-vec mult" << endl;;
        if(!zero_vector(matrix_message[i])) //cannot plainmult a vector of all zeroes
        {
            //cout << "size: " << p_matrix_diagonal_plain.size() << endl;
            //cout << "d_0: " << d_0 << endl << endl;
            
            evaluator->multiply_plain(query_enc, matrix[i], encrypted_sub_result);
            evaluator->relinearize_inplace(encrypted_sub_result, relin_keys);
            evaluator->rescale_to_next_inplace(encrypted_sub_result);
            evaluator->add_inplace(encrypted_result, encrypted_sub_result);
        }
        evaluator->rotate_vector(query_enc, 1, gal_keys, query_enc);
    }
    //cout << "after loop" << endl;
    Ciphertext encrypted_final_result = encrypted_result;

    
    
    for(int j = logd_1-1; j>=logd_0;j--)
    {
        evaluator->rotate_vector(encrypted_result, pow(2,j), gal_keys, encrypted_result);
        Ciphertext temp;
        evaluator->add(encrypted_final_result, encrypted_result, temp);
        encrypted_result = temp;
        encrypted_final_result = temp;
    }
    return encrypted_final_result;
}



void encrypted_feature_fusion_polynomial_approximation()
{

    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 16384;//8192;32768;//
    parms.set_poly_modulus_degree(poly_modulus_degree);
    //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
    //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60 }));
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 40, 40, 40, 40, 60 }));
    double scale = pow(2.0, 40);

    SEALContext context(parms);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    print_parameters(context);
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
    slot_count = 8;
    cout << "Number of slots: " << slot_count << endl;
    
    
    //set up coefficients for normalization
    size_t slots = encoder.slot_count();
    vector<double> a1, b1, c1, d1, a2, b2, c2;
    Plaintext a1_plain, b1_plain, c1_plain, d1_plain, a2_plain, b2_plain, c2_plain;
    
    
    //500-4000 train set
    //-6.96535417e+09  7.97134488e+06 -2.49368280e+03  2.78740519e-01
    //2.71838862e-02 -4.26441476e-12
    for(int i=0; i<slots; i++)
    {
        a1.push_back(-6.96535417e+09);
        b1.push_back(7.97134488e+06);
        c1.push_back(-2.49368280e+03);
        d1.push_back(2.78740519e-01);
        a2.push_back(2.71838862e-02);
        b2.push_back(-4.26441476e-12);
        //c2.push_back(7.50863559e-18);
    }

    
    //for(int i=0; i<slots; i++)
    //{
        //a1.push_back(4.03352862e+07);
        //b1.push_back(-1.15504744e+03);
        //c1.push_back(2.57773470e-01);
        //d1.push_back(-2.12343813e-05);
        //a2.push_back(-6.01341026e-01);
        //b2.push_back(1.60411938e-08);
        //c2.push_back(0);
    //}
    //[ 4.03352862e+07 -1.15504744e+03  2.57773470e-01 -2.12343813e-05
    //-6.01341026e-01  1.60411938e-08]
    //
    
    
    //[-9.47286767e+09  3.44595595e+04 -1.09943434e+01  1.14507908e-03
    // -1.47951062e+01 -1.56957587e-09]
    //encode coefficients for normalization now that encoder is defined
    encoder.encode(a1, scale, a1_plain);
    encoder.encode(b1, scale, b1_plain);
    encoder.encode(c1, scale, c1_plain);
    encoder.encode(d1, scale, d1_plain);
    encoder.encode(a2, scale, a2_plain);
    encoder.encode(b2, scale, b2_plain);
    encoder.encode(c2, scale, c2_plain);
    

    
    //read data file to get A values
    
    vector<vector<double>> A_message;
    vector<Plaintext> A_plain;
    vector<Ciphertext> A_enc;
    
    string row;
    ifstream A_file ("../A_values.txt");
    string A_values_str;
    int zeroes_to_add;
    int to_rotate = -1; //we need to know how much to rotate B by for concatenation
    
    
    //we can pack many vectors into a single ciphertext
    //each vector takes its size (slot_count) twice to allow for rotation to work
    int max_packed_words = slots / (2 * slot_count);
    //max_packed_words = 40;
    int max_words_packed_in_single_cipher = -1;
    int packed_words = 0;
    
    cout << "build A" << endl;
    vector<double> message_temp;
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
          row.push_back(' ');
          
          //vector<double> message_temp;
          //Plaintext plain_temp;
          //Ciphertext enc_temp;
          
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
          //cout << "zeroes to add " << zeroes_to_add << endl;
          for(int i = 0; i < zeroes_to_add; i++)
              message_temp.push_back(0.0);
          if(to_rotate == -1)
              to_rotate = -1 * (slot_count - zeroes_to_add);
          
          int to_replicate = slot_count;//message_temp.size();
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          
          //cout << "encode" << endl;
          packed_words++;
          if(packed_words >= max_packed_words)
          {
              if(packed_words>max_words_packed_in_single_cipher)
                  max_words_packed_in_single_cipher = packed_words;
              encoder.encode(message_temp, scale, plain_temp);
              
              //cout << "encrypt" << endl;
              encryptor.encrypt(plain_temp, enc_temp);
              
              //cout << "push back" << endl;
              A_enc.push_back(enc_temp);
              A_plain.push_back(plain_temp);
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
        
        //cout << "encrypt" << endl;
        encryptor.encrypt(plain_temp, enc_temp);
        
        //cout << "push back" << endl;
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
    
    ifstream B_file ("../B_values.txt");
    string B_values_str;
    
    cout << "build B" << endl;
    
    if (B_file.is_open())
    {
      while ( getline (B_file,row) )
      {
          //remove commas, source: https://stackoverflow.com/questions/20326356/how-to-remove-all-the-occurrences-of-a-char-in-c-string
          row.erase(remove(row.begin(), row.end(), ','), row.end());
          
          zeroes_to_add = slot_count;
          //strip off start and end brackets
          row = row.substr(1,B_values_str.length()-2);
          row.push_back(' ');
          
          //vector<double> message_temp;
          //Plaintext plain_temp;
          //Ciphertext enc_temp;
          
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
          //cout << message_temp.size() << endl;
          
          int to_replicate = slot_count;//message_temp.size();
          //cout << to_replicate << endl;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          
          packed_words++;
          if(packed_words >= max_packed_words)
          {
              //cout << "encode ";
              encoder.encode(message_temp, scale, plain_temp);
              //cout << "encoded";
              encryptor.encrypt(plain_temp, enc_temp);
              B_enc.push_back(enc_temp);
              B_plain.push_back(plain_temp);
              B_message.push_back(message_temp);
              message_temp.clear();
              packed_words = 0;
          }
          
      }
    if(message_temp.size() > 0)
    {
        encoder.encode(message_temp, scale, plain_temp);
        
        //cout << "encrypt" << endl;
        encryptor.encrypt(plain_temp, enc_temp);
        
        //cout << "push back" << endl;
        B_enc.push_back(enc_temp);
        B_plain.push_back(plain_temp);
        B_message.push_back(message_temp);
        message_temp.clear();
        packed_words = 0;
    }
        
        
    B_file.close();
    }
    
    
    
    cout << "B size: " << B_message.size() << endl;
    
    cout << "concatenate dataset" << endl;
    start = high_resolution_clock::now();
    
    //build encrypted dataset by rotating each query in B, adding together
    //this is effectively concatenation of each row of A and B
    cout << to_rotate << endl;
    vector<Ciphertext> enc_queries;
    for(int i = 0; i < A_enc.size(); i++)
    {
        Ciphertext query;
        Ciphertext B_temp = B_enc[i];
        //for(int j = 0; j < 5; j++)
            //evaluator.rotate_vector(B_temp, 1, gal_keys, B_temp);
        evaluator.rotate_vector(B_enc[i], to_rotate, gal_keys, B_temp);
        evaluator.add(A_enc[i], B_temp, query);
        enc_queries.push_back(query);
        
        Plaintext plain_result1;
        decryptor.decrypt(query, plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        //cout << "query: ";
        //print_vector(result1, 5, 7);
        //cout << result1.size() << endl;
    }
    
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to concatenate: " << duration.count() / 1000.0 << " milliseconds" << endl;
    
    //cout << "dataset built, here is a sample: ";
    
    //Plaintext test_plain;
    //vector<double> test_message;
    //decryptor.decrypt(enc_queries[0], test_plain);
    //encoder.decode(test_plain, test_message);
    //print_vector(test_message, 64, 3);
    
    
    vector<Ciphertext> p_matrix_diagonal;
    vector<Plaintext> p_matrix_diagonal_plain;
    vector<vector<double>> p_matrix_diagonal_message;
    p_matrix_diagonal.reserve(slot_count);
    
    cout << "building P" << endl;
    //read data file to get best P value
    //string row;
    ifstream P_file ("../diagonal_P_value_transpose_lambda=0.5_margin=0.5.txt");
    string P_values_str;
    message_temp.clear();
    packed_words = 0;
    if (P_file.is_open())
    {
      while ( getline (P_file,row) )
      {
          //vector<double> message_temp;
          //Plaintext plain_temp;
          //Ciphertext enc_temp;
          
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
              for(int i=0;i<to_replicate*2;i++)
                  message_temp.push_back(message_temp[i]);
          }
          //packed_words++;
          //if(packed_words >= max_packed_words)
          //{
              encoder.encode(message_temp, scale, plain_temp);
              encryptor.encrypt(plain_temp, enc_temp);
              p_matrix_diagonal.push_back(enc_temp);
              p_matrix_diagonal_plain.push_back(plain_temp);
              p_matrix_diagonal_message.push_back(message_temp);
              
              message_temp.clear();
              packed_words = 0;
          //}
          
      }
    //if(message_temp.size() > 0)
    //{
        //encoder.encode(message_temp, scale, plain_temp);
        //encryptor.encrypt(plain_temp, enc_temp);
        //p_matrix_diagonal.push_back(enc_temp);
        //p_matrix_diagonal_plain.push_back(plain_temp);
        //p_matrix_diagonal_message.push_back(message_temp);
        
        //message_temp.clear();
        //packed_words = 0;
    //}
    P_file.close();
    }
    //print_vector(p_matrix_diagonal_message[0], 32, 7);
    //cout << "p diag plain size: " << p_matrix_diagonal_plain.size() << endl;
    //cout << "done using new diag" << endl;
    
    //p matrix is d_0 by d_1
    //calculate later
    int d_0 = 2;
    int d_1 = 8;
    int logd_0 = 1;
    int logd_1 = 3;
    
    cout << "begin fusion" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    start = high_resolution_clock::now();
    
    vector<Ciphertext> enc_fusions;

    vector<double> message_zeroes;
    vector<double> message_ones;
    message_ones.clear();
    vector<double> message_zeroes2;
    Plaintext plain_zeroes;
    Plaintext plain_ones;
    Plaintext plain_zeroes2;

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
    //for(int c = 0; c < 2; c++)
    {
        Ciphertext encrypted_result;
        /*
        
        vector<double> message_zeroes;
        //vector<double> message_ones;
        message_ones.clear();
        vector<double> message_zeroes2;
        Plaintext plain_zeroes;
        //Plaintext plain_ones;
        Plaintext plain_zeroes2;
        //for (int i = 0; i < slot_count*2; i++)//doubled slot count
        for (int i = 0; i < slots; i++)
        {
            message_zeroes.push_back(0.0);
            message_ones.push_back(1.0);
            message_zeroes2.push_back(0.0);
        }
            
        encoder.encode(message_zeroes, scale, plain_zeroes);
        encoder.encode(message_ones, scale, plain_ones);
        encoder.encode(message_zeroes2, scale, plain_zeroes2);
         */
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
         
        //Ciphertext encrypted_final_result = hybrid_matmul(p_matrix_diagonal_plain, p_matrix_diagonal_message, enc_queries[c], d_0, d_1, &encryptor, &evaluator, relin_keys, gal_keys, plain_zeroes, plain_ones);
        enc_fusions.push_back(encrypted_final_result);
    }
    
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to fuse: " << duration.count() / 1000.0 << " milliseconds" << endl;
    
    
    //Normalization begins here////////////////////////////////////
    cout << "begin normalization" << endl;
    
    
    vector<Ciphertext> normalized_enc_fusions;
    
    start = high_resolution_clock::now();
    for(int i=0; i < enc_fusions.size(); i++)
    {
        Ciphertext cipher = enc_fusions[i];
        
        int vector_size = 2; // test value
        
        //Homormorphic inner product to find squared norm of input
        //mult depth = 1
        int iterations = log2(vector_size);
        Ciphertext squared_norm;
        evaluator.square(cipher, squared_norm);
        //evaluator.multiply(cipher, cipher, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        
        for(int i = 0; i < iterations; i++)
        {
            Ciphertext temp;
            evaluator.rotate_vector(squared_norm, pow(2, i), gal_keys, temp);
            evaluator.add(squared_norm, temp, squared_norm);
        }
        
        //end HE inner product
        
        //Calculate approximate inverse norm from squared norm
        Ciphertext result;
        
        Ciphertext x1, x2, x3, x4;
        Ciphertext squared_norm_down1, squared_norm_down2;
        Ciphertext enc_ones, enc_ones_down1, enc_ones_down2, enc_ones_down3;
        
        
        encryptor.encrypt(plain_ones, enc_ones);
        evaluator.multiply_plain(enc_ones, plain_ones, enc_ones_down1);
        evaluator.relinearize_inplace(enc_ones_down1, relin_keys);
        evaluator.rescale_to_next_inplace(enc_ones_down1);
        
        
        evaluator.multiply(enc_ones_down1, enc_ones_down1, enc_ones_down2);
        evaluator.relinearize_inplace(enc_ones_down2, relin_keys);
        evaluator.rescale_to_next_inplace(enc_ones_down2);
        
        evaluator.multiply(enc_ones_down2, enc_ones_down2, enc_ones_down3);
        evaluator.relinearize_inplace(enc_ones_down3, relin_keys);
        evaluator.rescale_to_next_inplace(enc_ones_down3);


        evaluator.multiply(squared_norm, enc_ones_down2, squared_norm_down1);
        evaluator.relinearize_inplace(squared_norm_down1, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm_down1);

        evaluator.multiply(squared_norm_down1, enc_ones_down3, squared_norm_down2);
        evaluator.relinearize_inplace(squared_norm_down2, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm_down2);

        
        //levels
        //x#_e = 9 //initially at least
        //enc_ones = 9
        //enc_ones_down1 = 8
        //enc_ones_down2 = 7
        //enc_ones_down3 = 6
        //squared_norm = 7
        //squared_norm_down1 = 6
        //result = 4

        evaluator.mod_switch_to_inplace(b1_plain, squared_norm.parms_id());
        b1_plain.scale() = pow(2,40);
        
        evaluator.multiply_plain(squared_norm, b1_plain, x2);
        evaluator.relinearize_inplace(x2, relin_keys);
        evaluator.rescale_to_next_inplace(x2);//x2 is at level 6

        evaluator.mod_switch_to_inplace(c1_plain, squared_norm.parms_id());
        c1_plain.scale() = pow(2,40);
        
        evaluator.multiply_plain(squared_norm, c1_plain, x3);
        evaluator.relinearize_inplace(x3, relin_keys);
        evaluator.rescale_to_next_inplace(x3);//x3 is at level 6

        evaluator.multiply(x3, squared_norm_down1, x3);
        evaluator.relinearize_inplace(x3, relin_keys);
        evaluator.rescale_to_next_inplace(x3);//x3 is at level 5
        
        evaluator.mod_switch_to_inplace(d1_plain, squared_norm.parms_id());
        d1_plain.scale() = pow(2,40);
        
        
        evaluator.multiply_plain(squared_norm, d1_plain, x4);
        evaluator.relinearize_inplace(x4, relin_keys);
        evaluator.rescale_to_next_inplace(x4);//x4 is at level 6
        evaluator.multiply(x4, squared_norm_down1, x4);
        evaluator.relinearize_inplace(x4, relin_keys);
        evaluator.rescale_to_next_inplace(x4);//x4 is at level 5
        evaluator.multiply(x4, squared_norm_down2, x4);
        evaluator.relinearize_inplace(x4, relin_keys);
        evaluator.rescale_to_next_inplace(x4);//x4 is at level 4
        
        //additional depth = 3
        //total depth = 4
        
        evaluator.mod_switch_to_inplace(a1_plain, x4.parms_id());
        evaluator.mod_switch_to_inplace(x2, x4.parms_id());
        evaluator.mod_switch_to_inplace(x3, x4.parms_id());
        a1_plain.scale() = pow(2,40);
        x2.scale() = pow(2,40);
        x3.scale() = pow(2,40);
        x4.scale() = pow(2,40);
        
        evaluator.add_plain(x2, a1_plain, result);
    
        evaluator.add_inplace(result, x3);
        evaluator.add_inplace(result, x4);
        
        
        evaluator.mod_switch_to_inplace(b2_plain, result.parms_id());
        b2_plain.scale() = pow(2,40);
        
        evaluator.multiply_plain(result, b2_plain, x2);
        evaluator.relinearize_inplace(x2, relin_keys);
        evaluator.rescale_to_next_inplace(x2);//x2 is at level 3
        
        
        
        
        //uncomment this section if c2 is not 0
        
        //evaluator.mod_switch_to_inplace(c2_plain, result.parms_id());
        //c2_plain.scale() = pow(2,40);
        //evaluator.multiply_plain(result, c2_plain, x3);
        //evaluator.relinearize_inplace(x3, relin_keys);
        //evaluator.rescale_to_next_inplace(x3);//x3 is at level 3
        //evaluator.mod_switch_to_inplace(result, x3.parms_id());
        //result.scale() = pow(2,40);
        //evaluator.multiply(x3, result, x3);
        //evaluator.relinearize_inplace(x3, relin_keys);
        //evaluator.rescale_to_next_inplace(x3);//x3 is at level 2
         
        
        //additional depth = 2
        //total depth = 6
        
        evaluator.mod_switch_to_inplace(a2_plain, x2.parms_id());
        a2_plain.scale() = x2.scale();
        
        evaluator.add_plain(x2, a2_plain, result);
        
        //uncomment if c2 is not 0
        
        //evaluator.mod_switch_to_inplace(result, x3.parms_id());
        //result.scale() = pow(2,40);
        //x3.scale() = pow(2,40);
        //evaluator.add_inplace(result, x3);//result is at level 2
         
        
        //result is our inverse norm
        
        //now we multiply the original vector by the inverse norm and return that result
        Ciphertext normalized_cipher;
        
        evaluator.mod_switch_to_inplace(cipher, result.parms_id());
        cipher.scale() = x2.scale();
        
        evaluator.multiply(cipher, result, normalized_cipher);
        evaluator.relinearize_inplace(normalized_cipher, relin_keys);
        evaluator.rescale_to_next_inplace(normalized_cipher);
        normalized_enc_fusions.push_back(normalized_cipher);
    }
    
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to normalize: " << duration.count() / 1000.0 << " milliseconds" << endl;
    cout << "normalization complete" << endl;
    
    //additional depth = 1
    //total depth = 7
    
    //END normalization
    
    //Plaintext plain_result1;
    //decryptor.decrypt(normalized_cipher, plain_result1);
    //vector<double> result1;
    //encoder.decode(plain_result1, result1);
    //print_vector(result1, 20, 7);
    
    
    
    
    
    
    for(int c = 0; c < normalized_enc_fusions.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(normalized_enc_fusions[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            double mag = 0;
            for(int j = slot_count*2*i; j < slot_count*2*i+2; j++)
            {
                //cout << result[j] << " ";
                mag+= pow(result[j],2);
            }
            mag = pow(mag,0.5);
            cout << mag << endl;
            //cout << endl;
        }
    }
    

}


void encrypted_feature_fusion_goldschmidt()
{
    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 32768;//8192;//16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
    //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60 }));
    //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 40, 40, 40, 40, 60 }));
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60 }));
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60 }));
    
    double scale = pow(2.0, 40);

    SEALContext context(parms);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    print_parameters(context);
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
    slot_count = 8;
    cout << "Number of slots: " << slot_count << endl;
    
    //set up constants for normalization
    size_t slots = encoder.slot_count();
    vector<double> constant_approx, neg_half, three_half, half, three_half_times_guess, two;
    Plaintext constant_approx_plain, neg_half_plain, three_half_plain, half_plain, three_half_times_guess_plain, two_plain;
    
    
    //500-4000 train set
    //-6.96535417e+09  7.97134488e+06 -2.49368280e+03  2.78740519e-01
    //2.71838862e-02 -4.26441476e-12
    for(int i=0; i<slots; i++)
    {
        //constant_approx.push_back(0.01);
        constant_approx.push_back(0.025);
        neg_half.push_back(-0.5);
        three_half.push_back(1.5);
        half.push_back(0.5);
        three_half_times_guess.push_back(0.0375);
        two.push_back(2.0);
    }

    //encode coefficients for normalization now that encoder is defined
    encoder.encode(constant_approx, scale, constant_approx_plain);
    encoder.encode(neg_half, scale, neg_half_plain);
    encoder.encode(three_half, scale, three_half_plain);
    encoder.encode(half, scale, half_plain);
    encoder.encode(three_half_times_guess, scale, three_half_times_guess_plain);
    encoder.encode(two, scale, two_plain);
    
    //read data file to get A values
    
    vector<vector<double>> A_message;
    vector<Plaintext> A_plain;
    vector<Ciphertext> A_enc;
    
    string row;
    ifstream A_file ("../A_values.txt");
    string A_values_str;
    int zeroes_to_add;
    int to_rotate = -1; //we need to know how much to rotate B by for concatenation
    
    
    //we can pack many vectors into a single ciphertext
    //each vector takes its size (slot_count) twice to allow for rotation to work
    int max_packed_words = slots / (2 * slot_count);
    //max_packed_words = 40;
    int max_words_packed_in_single_cipher = -1;
    int packed_words = 0;
    
    cout << "build A" << endl;
    vector<double> message_temp;
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
          row.push_back(' ');
          
          //vector<double> message_temp;
          //Plaintext plain_temp;
          //Ciphertext enc_temp;
          
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
          //cout << "zeroes to add " << zeroes_to_add << endl;
          for(int i = 0; i < zeroes_to_add; i++)
              message_temp.push_back(0.0);
          if(to_rotate == -1)
              to_rotate = -1 * (slot_count - zeroes_to_add);
          
          int to_replicate = slot_count;//message_temp.size();
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          
          //cout << "encode" << endl;
          packed_words++;
          if(packed_words >= max_packed_words)
          {
              if(packed_words>max_words_packed_in_single_cipher)
                  max_words_packed_in_single_cipher = packed_words;
              encoder.encode(message_temp, scale, plain_temp);
              
              //cout << "encrypt" << endl;
              encryptor.encrypt(plain_temp, enc_temp);
              
              //cout << "push back" << endl;
              A_enc.push_back(enc_temp);
              A_plain.push_back(plain_temp);
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
        
        //cout << "encrypt" << endl;
        encryptor.encrypt(plain_temp, enc_temp);
        
        //cout << "push back" << endl;
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
    
    ifstream B_file ("../B_values.txt");
    string B_values_str;
    
    cout << "build B" << endl;
    
    if (B_file.is_open())
    {
      while ( getline (B_file,row) )
      {
          //remove commas, source: https://stackoverflow.com/questions/20326356/how-to-remove-all-the-occurrences-of-a-char-in-c-string
          row.erase(remove(row.begin(), row.end(), ','), row.end());
          
          zeroes_to_add = slot_count;
          //strip off start and end brackets
          row = row.substr(1,B_values_str.length()-2);
          row.push_back(' ');
          
          //vector<double> message_temp;
          //Plaintext plain_temp;
          //Ciphertext enc_temp;
          
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
          //cout << message_temp.size() << endl;
          
          int to_replicate = slot_count;//message_temp.size();
          //cout << to_replicate << endl;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          
          packed_words++;
          if(packed_words >= max_packed_words)
          {
              //cout << "encode ";
              encoder.encode(message_temp, scale, plain_temp);
              //cout << "encoded";
              encryptor.encrypt(plain_temp, enc_temp);
              B_enc.push_back(enc_temp);
              B_plain.push_back(plain_temp);
              B_message.push_back(message_temp);
              message_temp.clear();
              packed_words = 0;
          }
          
      }
    if(message_temp.size() > 0)
    {
        encoder.encode(message_temp, scale, plain_temp);
        
        //cout << "encrypt" << endl;
        encryptor.encrypt(plain_temp, enc_temp);
        
        //cout << "push back" << endl;
        B_enc.push_back(enc_temp);
        B_plain.push_back(plain_temp);
        B_message.push_back(message_temp);
        message_temp.clear();
        packed_words = 0;
    }
        
        
    B_file.close();
    }
    
    
    
    cout << "B size: " << B_message.size() << endl;
    
    cout << "concatenate dataset" << endl;
    start = high_resolution_clock::now();
    
    //build encrypted dataset by rotating each query in B, adding together
    //this is effectively concatenation of each row of A and B
    cout << to_rotate << endl;
    vector<Ciphertext> enc_queries;
    for(int i = 0; i < A_enc.size(); i++)
    {
        Ciphertext query;
        Ciphertext B_temp = B_enc[i];
        //for(int j = 0; j < 5; j++)
            //evaluator.rotate_vector(B_temp, 1, gal_keys, B_temp);
        evaluator.rotate_vector(B_enc[i], to_rotate, gal_keys, B_temp);
        evaluator.add(A_enc[i], B_temp, query);
        enc_queries.push_back(query);
        
        Plaintext plain_result1;
        decryptor.decrypt(query, plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        //cout << "query: ";
        //print_vector(result1, 5, 7);
        //cout << result1.size() << endl;
    }
    
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to concatenate: " << duration.count() / 1000.0 << " milliseconds" << endl;
    
    //cout << "dataset built, here is a sample: ";
    
    //Plaintext test_plain;
    //vector<double> test_message;
    //decryptor.decrypt(enc_queries[0], test_plain);
    //encoder.decode(test_plain, test_message);
    //print_vector(test_message, 64, 3);
    
    
    vector<Ciphertext> p_matrix_diagonal;
    vector<Plaintext> p_matrix_diagonal_plain;
    vector<vector<double>> p_matrix_diagonal_message;
    p_matrix_diagonal.reserve(slot_count);
    
    cout << "building P" << endl;
    //read data file to get best P value
    //string row;
    
    vector<string> filenames{"../diagonal_P_value_transpose_lambda=0.5_margin=0.5.txt","../diagonal_random_P_value_transpose_indim=2_outdim=1.txt",
        "../diagonal_random_P_value_transpose_indim=4_outdim=2.txt","../diagonal_random_P_value_transpose_indim=8_outdim=4.txt",
        "../diagonal_random_P_value_transpose_indim=16_outdim=8.txt","../diagonal_random_P_value_transpose_indim=32_outdim=16.txt",
        "../diagonal_random_P_value_transpose_indim=64_outdim=32.txt","../diagonal_random_P_value_transpose_indim=128_outdim=64.txt"
    };
    int index_of_filenames = 0;
    //ifstream P_file ("../diagonal_P_value_transpose_lambda=0.5_margin=0.5.txt");
    ifstream P_file (filenames[index_of_filenames]);
    
    string P_values_str;
    message_temp.clear();
    packed_words = 0;
    if (P_file.is_open())
    {
      while ( getline (P_file,row) )
      {
          //vector<double> message_temp;
          //Plaintext plain_temp;
          //Ciphertext enc_temp;
          
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
              for(int i=0;i<to_replicate*2;i++)
                  message_temp.push_back(message_temp[i]);
          }
          //packed_words++;
          //if(packed_words >= max_packed_words)
          //{
              encoder.encode(message_temp, scale, plain_temp);
              encryptor.encrypt(plain_temp, enc_temp);
              p_matrix_diagonal.push_back(enc_temp);
              p_matrix_diagonal_plain.push_back(plain_temp);
              p_matrix_diagonal_message.push_back(message_temp);
              
              message_temp.clear();
              packed_words = 0;
          //}
          
      }
    //if(message_temp.size() > 0)
    //{
        //encoder.encode(message_temp, scale, plain_temp);
        //encryptor.encrypt(plain_temp, enc_temp);
        //p_matrix_diagonal.push_back(enc_temp);
        //p_matrix_diagonal_plain.push_back(plain_temp);
        //p_matrix_diagonal_message.push_back(message_temp);
        
        //message_temp.clear();
        //packed_words = 0;
    //}
    P_file.close();
    }
    //print_vector(p_matrix_diagonal_message[0], 32, 7);
    //cout << "p diag plain size: " << p_matrix_diagonal_plain.size() << endl;
    //cout << "done using new diag" << endl;
    
    //p matrix is d_0 by d_1
    //calculate later
    int d_0 = 2;
    int d_1 = 8;
    int logd_0 = 1;
    int logd_1 = 3;
    
    cout << "begin fusion" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    start = high_resolution_clock::now();
    
    vector<Ciphertext> enc_fusions;
    vector<double> message_ones;
    Plaintext plain_ones;
    for(int c = 0; c < enc_queries.size(); c++)
    //for(int c = 0; c < 2; c++)
    {
        Ciphertext encrypted_result;
        vector<double> message_zeroes;
        //vector<double> message_ones;
        message_ones.clear();
        vector<double> message_zeroes2;
        Plaintext plain_zeroes;
        //Plaintext plain_ones;
        Plaintext plain_zeroes2;
        //for (int i = 0; i < slot_count*2; i++)//doubled slot count
        for (int i = 0; i < slots; i++)
        {
            message_zeroes.push_back(0.0);
            message_ones.push_back(1.0);
            message_zeroes2.push_back(0.0);
        }
            
        encoder.encode(message_zeroes, scale, plain_zeroes);
        encoder.encode(message_ones, scale, plain_ones);
        encoder.encode(message_zeroes2, scale, plain_zeroes2);
        encryptor.encrypt(plain_zeroes, encrypted_result);
        

        
        evaluator.multiply_plain(encrypted_result, plain_ones, encrypted_result);
        evaluator.relinearize_inplace(encrypted_result, relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result); //level of encrypted_result must match sub result, which will be rescaled once
        
        Ciphertext encrypted_sub_result;
        
        for(int i = 0; i < d_0; i++)
        {
            cout << "iteration "<<i<< " of mat-vec mult";
            if(!zero_vector(p_matrix_diagonal_message[i])) //cannot plainmult a vector of all zeroes
            {

                evaluator.multiply_plain(enc_queries[c], p_matrix_diagonal_plain[i], encrypted_sub_result);
                evaluator.relinearize_inplace(encrypted_sub_result, relin_keys);
                evaluator.rescale_to_next_inplace(encrypted_sub_result);
                evaluator.add_inplace(encrypted_result, encrypted_sub_result);
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
    
    
    //Normalization begins here////////////////////////////////////
    cout << "begin normalization" << endl;
    
    
    vector<Ciphertext> normalized_enc_fusions;
    
    start = high_resolution_clock::now();
    for(int i=0; i < enc_fusions.size(); i++)
    {
        Ciphertext cipher = enc_fusions[i];
        
        int vector_size = 2; // test value
        
        //Homormorphic inner product to find squared norm of input
        //mult depth = 1
        int iterations = log2(vector_size);
        Ciphertext squared_norm;
        evaluator.square(cipher, squared_norm);
        //evaluator.multiply(cipher, cipher, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        
        for(int i = 0; i < iterations; i++)
        {
            Ciphertext temp;
            evaluator.rotate_vector(squared_norm, pow(2, i), gal_keys, temp);
            evaluator.add(squared_norm, temp, squared_norm);
        }
        
        //end HE inner product
        
        //Calculate approximate inverse norm from squared norm
        Ciphertext result;
        
        Ciphertext x1, x2, x3, x4;
        Ciphertext squared_norm_down1, squared_norm_down2;
        Ciphertext enc_ones, enc_ones_down1, enc_ones_down2, enc_ones_down3;
        
        
        encryptor.encrypt(plain_ones, enc_ones);
        evaluator.multiply_plain(enc_ones, plain_ones, enc_ones_down1);
        evaluator.relinearize_inplace(enc_ones_down1, relin_keys);
        evaluator.rescale_to_next_inplace(enc_ones_down1);
        
        
        evaluator.multiply(enc_ones_down1, enc_ones_down1, enc_ones_down2);
        evaluator.relinearize_inplace(enc_ones_down2, relin_keys);
        evaluator.rescale_to_next_inplace(enc_ones_down2);
        
        evaluator.multiply(enc_ones_down2, enc_ones_down2, enc_ones_down3);
        evaluator.relinearize_inplace(enc_ones_down3, relin_keys);
        evaluator.rescale_to_next_inplace(enc_ones_down3);


        evaluator.multiply(squared_norm, enc_ones_down2, squared_norm_down1);
        evaluator.relinearize_inplace(squared_norm_down1, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm_down1);

        evaluator.multiply(squared_norm_down1, enc_ones_down3, squared_norm_down2);
        evaluator.relinearize_inplace(squared_norm_down2, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm_down2);

        
        //levels
        //x#_e = 9 //initially at least
        //enc_ones = 9
        //enc_ones_down1 = 8
        //enc_ones_down2 = 7
        //enc_ones_down3 = 6
        //squared_norm = 7
        //squared_norm_down1 = 6
        //result = 4

        //evaluator.mod_switch_to_inplace(b1_plain, squared_norm.parms_id());
        //b1_plain.scale() = pow(2,40);
        
        //evaluator.multiply_plain(squared_norm, b1_plain, x2);
        //evaluator.relinearize_inplace(x2, relin_keys);
        //evaluator.rescale_to_next_inplace(x2);//x2 is at level 6

        
        //constant_approx_plain, neg_half_plain, three_half_plain, half_plain;
        //squared_norm
        //result
        
        //fast inverse square root, one iteration
        
        Ciphertext guess, temp;
        
        evaluator.mod_switch_to_inplace(constant_approx_plain, squared_norm.parms_id());
        constant_approx_plain.scale() = pow(2,40);
        evaluator.multiply_plain(squared_norm, constant_approx_plain, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        cout <<"first mul done"<<endl;
        evaluator.mod_switch_to_inplace(constant_approx_plain, squared_norm.parms_id());
        constant_approx_plain.scale() = pow(2,40);
        evaluator.multiply_plain(squared_norm, constant_approx_plain, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        cout <<"second mul done"<<endl;
        
        evaluator.mod_switch_to_inplace(neg_half_plain, squared_norm.parms_id());
        neg_half_plain.scale() = pow(2,40);
        evaluator.multiply_plain(squared_norm, neg_half_plain, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        cout <<"third mul done"<<endl;
        
        evaluator.mod_switch_to_inplace(constant_approx_plain, squared_norm.parms_id());
        constant_approx_plain.scale() = pow(2,40);
        evaluator.multiply_plain(squared_norm, constant_approx_plain, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        cout <<"foruth mul done"<<endl;
        

        
        evaluator.mod_switch_to_inplace(three_half_times_guess_plain, squared_norm.parms_id());
        three_half_times_guess_plain.scale() = pow(2,40);
        squared_norm.scale() = pow(2,40);
        //cout << temp.scale() << endl;
        evaluator.add_plain(squared_norm, three_half_times_guess_plain, guess);
        //evaluator.relinearize_inplace(guess, relin_keys);
        //evaluator.rescale_to_next_inplace(guess);
        cout <<"first add done"<<endl;
        //fast inv square root end
        
        Plaintext plain_result1;
        decryptor.decrypt(guess, plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        print_vector(result1, 10, 7);
        
        
        //goldschmidts begin
        Ciphertext x, h, r;
        int iterations_gold = 4;
        
        
        evaluator.mod_switch_to_inplace(squared_norm_down1, guess.parms_id());
        squared_norm_down1.scale() = pow(2,40);
        evaluator.multiply(squared_norm_down1, guess, x);
        evaluator.relinearize_inplace(x, relin_keys);
        evaluator.rescale_to_next_inplace(x);

        evaluator.mod_switch_to_inplace(half_plain, guess.parms_id());
        half_plain.scale() = pow(2,40);
        evaluator.multiply_plain(guess, half_plain, h);
        evaluator.relinearize_inplace(h, relin_keys);
        evaluator.rescale_to_next_inplace(h);
        
        for(int i = 0; i < iterations_gold; i++)
        {

            //cout << "time for first g iter" << endl;
            cout << "goldschmidt iter: " << i << endl;
            
            evaluator.multiply(x, h, temp);
            evaluator.relinearize_inplace(temp, relin_keys);
            evaluator.rescale_to_next_inplace(temp);
            
            evaluator.negate_inplace(temp);
            //cout << "negated" << endl;
            
            cout <<"!" << endl;
            evaluator.mod_switch_to_inplace(half_plain, temp.parms_id());
            half_plain.scale() = pow(2,40);
            temp.scale() = pow(2,40);
            evaluator.add_plain(temp, half_plain, r);
            //cout << "added" << endl;
            
            cout <<"!" << endl;
            evaluator.mod_switch_to_inplace(x, r.parms_id());
            x.scale() = pow(2,40);
            evaluator.multiply(x, r, temp);
            evaluator.relinearize_inplace(temp, relin_keys);
            evaluator.rescale_to_next_inplace(temp);
            
            cout <<"!" << endl;
            evaluator.mod_switch_to_inplace(x, temp.parms_id());
            x.scale() = pow(2,40);
            temp.scale() = pow(2,40);
            evaluator.add(x, temp, x);
            //cout << "line 19 done" << endl;
            
            cout <<"!" << endl;
            evaluator.mod_switch_to_inplace(h, r.parms_id());
            h.scale() = pow(2,40);
            evaluator.multiply(h, r, temp);
            evaluator.relinearize_inplace(temp, relin_keys);
            evaluator.rescale_to_next_inplace(temp);
            
            cout <<"!" << endl;
            evaluator.mod_switch_to_inplace(h, temp.parms_id());
            h.scale() = pow(2,40);
            temp.scale() = pow(2,40);
            evaluator.add(h, temp, h);
            //cout << "line 20 done" << endl;
        }
        
        
        
        evaluator.add(h, h, guess);
        //goldschmidts end
        
        //now we multiply the original vector by the inverse norm and return that result
        Ciphertext normalized_cipher;
        
        evaluator.mod_switch_to_inplace(cipher, guess.parms_id());
        cipher.scale() = guess.scale();
        
        evaluator.multiply(cipher, guess, normalized_cipher);
        evaluator.relinearize_inplace(normalized_cipher, relin_keys);
        evaluator.rescale_to_next_inplace(normalized_cipher);
        normalized_enc_fusions.push_back(normalized_cipher);
    }
    
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to normalize: " << duration.count() / 1000.0 << " milliseconds" << endl;
    cout << "normalization complete" << endl;
    
    //additional depth = 1
    //total depth = 7
    
    //END normalization
    
    //Plaintext plain_result1;
    //decryptor.decrypt(normalized_cipher, plain_result1);
    //vector<double> result1;
    //encoder.decode(plain_result1, result1);
    //print_vector(result1, 20, 7);
    
    
    
    
    vector<int> labels;
    for(int i=0; i<4; i++)
    {
        for(int j=0;j<10;j++)
            labels.push_back(i);
    }
    
    
    for(int c = 0; c < normalized_enc_fusions.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(normalized_enc_fusions[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            double mag = 0;
            for(int j = slot_count*2*i; j < slot_count*2*i+2; j++)
            {
                cout << result[j];
                if(j != slot_count*2*i+2-1)
                    cout << " ";
                mag+= pow(result[j],2);
            }
            mag = pow(mag,0.5);
            //cout << mag << endl;
            cout << ";" << labels[i*(c+1)] << endl;
        }
    }
    

}


void encrypted_feature_fusion_vector_rows()
{

    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 8192;//16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 60 }));

    double scale = pow(2.0, 40);

    SEALContext context(parms);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    print_parameters(context);
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
    slot_count = 8;
    cout << "Number of slots: " << slot_count << endl;
    size_t slots = encoder.slot_count();
    
    
    vector<double> query_message;
    Plaintext query_plain;
    Ciphertext query_enc;
    vector<Ciphertext> enc_queries;
    
    
    string row;
    ifstream A_file ("../../data/A_values_transpose.txt");
    string A_values_str;
    
    
    vector<double> message_temp;
    Plaintext plain_temp;
    Ciphertext enc_temp;
    
    if (A_file.is_open())
    {
      cout << "build A" << endl;
      while ( getline (A_file,row) )
      {
          //remove commas, source: https://stackoverflow.com/questions/20326356/how-to-remove-all-the-occurrences-of-a-char-in-c-string
          row.erase(remove(row.begin(), row.end(), ','), row.end());
          
          //zeroes_to_add = slot_count;
          //strip off start and end brackets
          row = row.substr(1,A_values_str.length()-2);
          row.push_back(' ');
          size_t pos = 0;
          string A_sub_value;
          while((pos = row.find(' ')) != -1)
          {
              //zeroes_to_add--;
              A_sub_value = row.substr(0, pos-1);
              double A_sub_value_final = stod(A_sub_value);
              message_temp.push_back(A_sub_value_final);
              row.erase(0, pos+1);
          }
          encoder.encode(message_temp, scale, plain_temp);
          encryptor.encrypt(plain_temp, enc_temp);
          enc_queries.push_back(enc_temp);
          message_temp.clear();
      }
    A_file.close();
    }
    
    ifstream B_file ("../../data/B_values_transpose.txt");
    string A_values_str;
    
    vector<double> message_temp;
    Plaintext plain_temp;
    Ciphertext enc_temp;
    
    if (B_file.is_open())
    {
      cout << "build B" << endl;
      while ( getline (A_file,row) )
      {
          //remove commas, source: https://stackoverflow.com/questions/20326356/how-to-remove-all-the-occurrences-of-a-char-in-c-string
          row.erase(remove(row.begin(), row.end(), ','), row.end());
          
          //zeroes_to_add = slot_count;
          //strip off start and end brackets
          row = row.substr(1,A_values_str.length()-2);
          row.push_back(' ');
          size_t pos = 0;
          string B_sub_value;
          while((pos = row.find(' ')) != -1)
          {
              //zeroes_to_add--;
              B_sub_value = row.substr(0, pos-1);
              double B_sub_value_final = stod(B_sub_value);
              message_temp.push_back(B_sub_value_final);
              row.erase(0, pos+1);
          }
          encoder.encode(message_temp, scale, plain_temp);
          encryptor.encrypt(plain_temp, enc_temp);
          enc_queries.push_back(enc_temp);
          message_temp.clear();
      }
    B_file.close();
    }
    
    
    /*
    for(int i=0; i<slots; i++)
    {
        query_message.push_back(0.123);
    }
    encoder.encode(query_message, scale, query_plain);
    encryptor.encrypt(query_plain, query_enc);
    */
    
    //vector<string> filenames{"../data/random_P_value_transpose_indim=2_outdim=1.txt",
        //"../data/random_P_value_transpose_indim=4_outdim=1.txt","../data/random_P_value_transpose_indim=8_outdim=1.txt",
        //"../data/random_P_value_transpose_indim=16_outdim=1.txt","../data/random_P_value_transpose_indim=32_outdim=1.txt",
        //"../data/random_P_value_transpose_indim=64_outdim=1.txt","../data/random_P_value_transpose_indim=128_outdim=1.txt"
    //};
    vector<string> filenames{"../../data/best_P_value_transpose_lambda=0.5_margin=0.5.txt"};
    
    ///
    vector<double> message_zeroes;
    vector<double> message_ones;
    message_ones.clear();
    vector<double> message_zeroes2;
    Plaintext plain_zeroes;
    Plaintext plain_ones;
    Plaintext plain_zeroes2;
    //for (int i = 0; i < slot_count*2; i++)//doubled slot count
    for (int i = 0; i < slots; i++)
    {
        message_zeroes.push_back(0.0);
        message_ones.push_back(1.0);
        message_zeroes2.push_back(0.0);
    }
    encoder.encode(message_zeroes, scale, plain_zeroes);
    encoder.encode(message_ones, scale, plain_ones);
    encoder.encode(message_zeroes2, scale, plain_zeroes2);
    ///
    
    for(int q = 0; q < 1; q++)
    {
        cout << pow(2,q+1) << endl;
        vector<double> message_temp;
        Plaintext plain_temp;
        Ciphertext enc_temp;
        
        
        vector<Ciphertext> p_matrix;
        vector<Plaintext> p_matrix_plain;
        vector<vector<double>> p_matrix_message;
        p_matrix.reserve(slot_count);
        
        cout << "building P" << endl;
        int index_of_filenames = q;
        ifstream P_file (filenames[index_of_filenames]);
        
        string P_values_str;
        //string row;
        int packed_words = 0;
        int max_packed_words = 1;
        int max_dim = 128;
        if (P_file.is_open())
        {
          while ( getline (P_file,row) )
          {
              size_t pos = 0;
              string P_sub_value;
              
              //format of row-order files is a little different
              row = row.substr(2,row.size()-3);
              row = row + " ";
              
              while((pos = row.find(' ')) != -1)
              {
                  P_sub_value = row.substr(0, pos-1);
                  double P_sub_value_final = stod(P_sub_value);
                  for(int i=0;i<max_dim;i++)
                      message_temp.push_back(P_sub_value_final);
                  row.erase(0, pos+1);
                  
                  encoder.encode(message_temp, scale, plain_temp);
                  encryptor.encrypt(plain_temp, enc_temp);
                  p_matrix.push_back(enc_temp);
                  p_matrix_plain.push_back(plain_temp);
                  p_matrix_message.push_back(message_temp);
                  
                  message_temp.clear();
              }
              
              //int to_replicate = message_temp.size();
              //for(int i=0;i<to_replicate;i++)
                  //message_temp.push_back(message_temp[i]);
              
              //for(int i=0;i<(max_packed_words-1);i++)
              //{
                  //for(int i=0;i<to_replicate*2;i++)
                      //message_temp.push_back(message_temp[i]);
              //}

              
              //packed_words = 0;
          }

        P_file.close();
        }

        //p matrix is d_0 by d_1
        //calculate later
        int d_0 = 2;//out dimension
        int d_1 = 8;//in dimension
        int logd_0 = 1;
        int logd_1 = 3;
        
        cout << "begin fusion" << endl;
        //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
        
        
        //vector<Ciphertext> enc_fusions;
        
        start = high_resolution_clock::now();

        vector<Ciphertext> enc_fusions;
        Ciphertext encrypted_result;
        
        //encryptor.encrypt(plain_zeroes, encrypted_result);
        
        //evaluator.multiply_plain(encrypted_result, plain_ones, encrypted_result);
        //evaluator.relinearize_inplace(encrypted_result, relin_keys);
        //evaluator.rescale_to_next_inplace(encrypted_result); //level of encrypted_result must match sub result, which will be rescaled once
        
        for(int i=0; i<d_0; i++)
        {
            for(int j=0; j<d_1; j++)
            {
                Ciphertext encrypted_sub_result;

                //cout << "iteration " << i << " of mat-vec mult" << endl;;
                if(!zero_vector(p_matrix_message[i])) //cannot plainmult a vector of all zeroes
                {
                    //cout << "size: " << p_matrix_diagonal_plain.size() << endl;
                    //cout << "d_0: " << d_0 << endl << endl;
                    
                    evaluator.multiply_plain(enc_queries[j], p_matrix_plain[j+i*d_1], encrypted_sub_result);
                    evaluator.relinearize_inplace(encrypted_sub_result, relin_keys);
                    evaluator.rescale_to_next_inplace(encrypted_sub_result);
                    if(j!=0)
                        evaluator.add_inplace(encrypted_result, encrypted_sub_result);
                    else
                        encrypted_result = encrypted_sub_result;
                }
                
            }
            enc_fusions.push_back(encrypted_result);
        }
        
        //Ciphertext encrypted_final_result = hybrid_matmul(p_matrix_diagonal_plain, p_matrix_diagonal_message, query_enc, d_0, d_1, &encryptor, &evaluator, relin_keys, gal_keys, plain_zeroes, plain_ones);
        //enc_fusions.push_back(encrypted_final_result);

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "time to fuse: " << duration.count() / 1000.0 << " milliseconds" << endl;
        
        //Plaintext plain_result1;
        //decryptor.decrypt(enc_fusions[0], plain_result1);
        //vector<double> result1;
        //encoder.decode(plain_result1, result1);
        //print_vector(result1, 3, 7);
        
        cout << endl;
    }
}


void Matrix_Vector_Multiplication_Diagonal_Test()
{

    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 8192;//16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 60 }));
    
    double scale = pow(2.0, 40);

    SEALContext context(parms);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    print_parameters(context);
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
    slot_count = 8;
    cout << "Number of slots: " << slot_count << endl;
    size_t slots = encoder.slot_count();
    
    
    
    //read data file to get A values
    
    vector<double> query_message;
    Plaintext query_plain;
    Ciphertext query_enc;
    
    for(int i=0; i<slots; i++)
    {
        query_message.push_back(0.123);
    }
    encoder.encode(query_message, scale, query_plain);
    encryptor.encrypt(query_plain, query_enc);
    
    
    vector<string> filenames{"../diagonal_random_P_value_transpose_indim=128_outdim=1.txt",
        "../diagonal_random_P_value_transpose_indim=128_outdim=2.txt","../diagonal_random_P_value_transpose_indim=128_outdim=4.txt",
        "../diagonal_random_P_value_transpose_indim=128_outdim=8.txt","../diagonal_random_P_value_transpose_indim=128_outdim=16.txt",
        "../diagonal_random_P_value_transpose_indim=128_outdim=32.txt","../diagonal_random_P_value_transpose_indim=128_outdim=64.txt"
    };
    
    ///
    vector<double> message_zeroes;
    vector<double> message_ones;
    message_ones.clear();
    vector<double> message_zeroes2;
    Plaintext plain_zeroes;
    Plaintext plain_ones;
    Plaintext plain_zeroes2;
    //for (int i = 0; i < slot_count*2; i++)//doubled slot count
    for (int i = 0; i < slots; i++)
    {
        message_zeroes.push_back(0.0);
        message_ones.push_back(1.0);
        message_zeroes2.push_back(0.0);
    }
    encoder.encode(message_zeroes, scale, plain_zeroes);
    encoder.encode(message_ones, scale, plain_ones);
    encoder.encode(message_zeroes2, scale, plain_zeroes2);
    ///
    
    for(int q = 0; q < 7; q++)
    {
        cout << pow(2,q) << endl;
        vector<double> message_temp;
        Plaintext plain_temp;
        Ciphertext enc_temp;
        
        
        vector<Ciphertext> p_matrix_diagonal;
        vector<Plaintext> p_matrix_diagonal_plain;
        vector<vector<double>> p_matrix_diagonal_message;
        p_matrix_diagonal.reserve(slot_count);
        
        cout << "building P" << endl;
        int index_of_filenames = q;
        ifstream P_file (filenames[index_of_filenames]);
        
        string P_values_str;
        string row;
        int packed_words = 0;
        int max_packed_words = 1;
        if (P_file.is_open())
        {
          while ( getline (P_file,row) )
          {
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
                  for(int i=0;i<to_replicate*2;i++)
                      message_temp.push_back(message_temp[i]);
              }

              encoder.encode(message_temp, scale, plain_temp);
              encryptor.encrypt(plain_temp, enc_temp);
              p_matrix_diagonal.push_back(enc_temp);
              p_matrix_diagonal_plain.push_back(plain_temp);
              p_matrix_diagonal_message.push_back(message_temp);
              
              message_temp.clear();
              packed_words = 0;
          }

        P_file.close();
        }

        //p matrix is d_0 by d_1
        //calculate later
        int d_0 = pow(2,q);
        int d_1 = 128;
        int logd_0 = q;
        int logd_1 = 7;
        
        cout << "begin fusion" << endl;
        //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
        
        
        vector<Ciphertext> enc_fusions;
        Ciphertext enc_fusion;
        
        start = high_resolution_clock::now();

        Ciphertext encrypted_final_result = hybrid_matmul(p_matrix_diagonal_plain, p_matrix_diagonal_message, query_enc, d_0, d_1, &encryptor, &evaluator, relin_keys, gal_keys, plain_zeroes, plain_ones);
        //enc_fusions.push_back(encrypted_final_result);

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "time to fuse: " << duration.count() / 1000.0 << " milliseconds" << endl;
        
        //Plaintext plain_result1;
        //decryptor.decrypt(enc_fusions[0], plain_result1);
        //vector<double> result1;
        //encoder.decode(plain_result1, result1);
        //print_vector(result1, 3, 7);
        
        cout << endl;
    }
}


void Matrix_Vector_Multiplication_Vector_Rows_Test()
{

    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 8192;//16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 60 }));

    double scale = pow(2.0, 40);

    SEALContext context(parms);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    print_parameters(context);
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
    slot_count = 8;
    cout << "Number of slots: " << slot_count << endl;
    size_t slots = encoder.slot_count();
    
    
    vector<double> query_message;
    Plaintext query_plain;
    Ciphertext query_enc;
    //vector<Ciphertext> enc_queries;
    
    for(int i=0; i<slots; i++)
    {
        query_message.push_back(0.123);
    }
    encoder.encode(query_message, scale, query_plain);
    encryptor.encrypt(query_plain, query_enc);
    
    
    vector<string> filenames{"../data/random_P_value_transpose_indim=2_outdim=1.txt",
        "../data/random_P_value_transpose_indim=4_outdim=1.txt","../data/random_P_value_transpose_indim=8_outdim=1.txt",
        "../data/random_P_value_transpose_indim=16_outdim=1.txt","../data/random_P_value_transpose_indim=32_outdim=1.txt",
        "../data/random_P_value_transpose_indim=64_outdim=1.txt","../data/random_P_value_transpose_indim=128_outdim=1.txt"
    };
    
    ///
    vector<double> message_zeroes;
    vector<double> message_ones;
    message_ones.clear();
    vector<double> message_zeroes2;
    Plaintext plain_zeroes;
    Plaintext plain_ones;
    Plaintext plain_zeroes2;
    //for (int i = 0; i < slot_count*2; i++)//doubled slot count
    for (int i = 0; i < slots; i++)
    {
        message_zeroes.push_back(0.0);
        message_ones.push_back(1.0);
        message_zeroes2.push_back(0.0);
    }
    encoder.encode(message_zeroes, scale, plain_zeroes);
    encoder.encode(message_ones, scale, plain_ones);
    encoder.encode(message_zeroes2, scale, plain_zeroes2);
    ///
    
    for(int q = 0; q < 7; q++)
    {
        cout << pow(2,q+1) << endl;
        vector<double> message_temp;
        Plaintext plain_temp;
        Ciphertext enc_temp;
        
        
        vector<Ciphertext> p_matrix;
        vector<Plaintext> p_matrix_plain;
        vector<vector<double>> p_matrix_message;
        p_matrix.reserve(slot_count);
        
        cout << "building P" << endl;
        int index_of_filenames = q;
        ifstream P_file (filenames[index_of_filenames]);
        
        string P_values_str;
        string row;
        int packed_words = 0;
        int max_packed_words = 1;
        int max_dim = 128;
        if (P_file.is_open())
        {
          while ( getline (P_file,row) )
          {
              size_t pos = 0;
              string P_sub_value;
              
              //format of row-order files is a little different
              row = row.substr(2,row.size()-3);
              row = row + " ";
              
              while((pos = row.find(' ')) != -1)
              {
                  P_sub_value = row.substr(0, pos-1);
                  double P_sub_value_final = stod(P_sub_value);
                  for(int i=0;i<max_dim;i++)
                      message_temp.push_back(P_sub_value_final);
                  row.erase(0, pos+1);
                  
                  encoder.encode(message_temp, scale, plain_temp);
                  encryptor.encrypt(plain_temp, enc_temp);
                  p_matrix.push_back(enc_temp);
                  p_matrix_plain.push_back(plain_temp);
                  p_matrix_message.push_back(message_temp);
                  
                  message_temp.clear();
              }
              
              //int to_replicate = message_temp.size();
              //for(int i=0;i<to_replicate;i++)
                  //message_temp.push_back(message_temp[i]);
              
              //for(int i=0;i<(max_packed_words-1);i++)
              //{
                  //for(int i=0;i<to_replicate*2;i++)
                      //message_temp.push_back(message_temp[i]);
              //}

              
              //packed_words = 0;
          }

        P_file.close();
        }

        //p matrix is d_0 by d_1
        //calculate later
        int d_0 = 1;//out dimension
        int d_1 = pow(2,q);//in dimension
        int logd_0 = 0;
        int logd_1 = q;
        
        cout << "begin fusion" << endl;
        //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
        
        
        //vector<Ciphertext> enc_fusions;
        
        start = high_resolution_clock::now();

        vector<Ciphertext> enc_fusions;
        Ciphertext encrypted_result;
        
        //encryptor.encrypt(plain_zeroes, encrypted_result);
        
        //evaluator.multiply_plain(encrypted_result, plain_ones, encrypted_result);
        //evaluator.relinearize_inplace(encrypted_result, relin_keys);
        //evaluator.rescale_to_next_inplace(encrypted_result); //level of encrypted_result must match sub result, which will be rescaled once
        
        for(int i=0; i<d_0; i++)
        {
            for(int j=0; j<d_1; j++)
            {
                Ciphertext encrypted_sub_result;

                //cout << "iteration " << i << " of mat-vec mult" << endl;;
                if(!zero_vector(p_matrix_message[i])) //cannot plainmult a vector of all zeroes
                {
                    //cout << "size: " << p_matrix_diagonal_plain.size() << endl;
                    //cout << "d_0: " << d_0 << endl << endl;
                    
                    evaluator.multiply_plain(query_enc, p_matrix_plain[j+i*d_1], encrypted_sub_result);
                    evaluator.relinearize_inplace(encrypted_sub_result, relin_keys);
                    evaluator.rescale_to_next_inplace(encrypted_sub_result);
                    if(j!=0)
                        evaluator.add_inplace(encrypted_result, encrypted_sub_result);
                    else
                        encrypted_result = encrypted_sub_result;
                }
                
            }
            enc_fusions.push_back(encrypted_result);
        }
        
        //Ciphertext encrypted_final_result = hybrid_matmul(p_matrix_diagonal_plain, p_matrix_diagonal_message, query_enc, d_0, d_1, &encryptor, &evaluator, relin_keys, gal_keys, plain_zeroes, plain_ones);
        //enc_fusions.push_back(encrypted_final_result);

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "time to fuse: " << duration.count() / 1000.0 << " milliseconds" << endl;
        
        //Plaintext plain_result1;
        //decryptor.decrypt(enc_fusions[0], plain_result1);
        //vector<double> result1;
        //encoder.decode(plain_result1, result1);
        //print_vector(result1, 3, 7);
        
        cout << endl;
    }
}

void encrypted_feature_fusion()
{
    //Matrix_Vector_Multiplication_Diagonal_Test();
    //Matrix_Vector_Multiplication_Vector_Rows_Test();
    //encrypted_feature_fusion_polynomial_approximation();
    //encrypted_feature_fusion_goldschmidt();
    encrypted_feature_fusion_vector_rows();
}
