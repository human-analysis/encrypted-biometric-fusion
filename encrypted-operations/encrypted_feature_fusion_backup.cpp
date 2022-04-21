// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "examples.h"
#include <cmath>
#include <chrono>
#include <algorithm>

#include <filesystem>
#include <unistd.h>
#include <stdlib.h>

#include <string>

#include <fstream>

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




void encrypted_feature_fusion_polynomial_approximation_arbitrary(string P_file_name_in, string outfile_name, string normalized_outfile_name, int degree)
{

    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 16384;//8192;32768;//
    poly_modulus_degree = 32768;//
    parms.set_poly_modulus_degree(poly_modulus_degree);

    //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 40, 40, 40, 40, 40, 40, 40, 40, 50 })); //degree=4
    
    
    
    
    if(degree==1)
    {
        poly_modulus_degree = 16384;//
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 40, 40, 40, 40, 40, 50 })); //degree=2
    }
    else if(degree==2)
    {
        //poly_modulus_degree = 16384;//
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 40, 40, 40, 40, 40, 40, 50 })); //degree=2
        //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 60, 60, 60, 60, 60, 60, 60 })); //degree=2
    }
    else if (degree==3)
    {
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 40, 40, 40, 40, 40, 40, 40, 50 })); //degree=3
    }
    else if (degree==6)
    {
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50 })); //degree=6
    }
    
    
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
    
    
    slot_count = 1024;
    cout << "Number of slots: " << slot_count << endl;
    
    
    //set up coefficients for normalization
    size_t slots = encoder.slot_count();
    vector<double> a1, b1, c1, d1, a2, b2, c2, d2;
    Plaintext a1_plain, b1_plain, c1_plain, d1_plain, a2_plain, b2_plain, c2_plain, d2_plain;
    
    vector<double> tenth;
    Plaintext tenth_plain;
    
    
    for(int i=0; i<slots; i++)
    {
        /*
        //[ [0.83145916 -3.07507111  5.3092142  -4.42648824] 1.33883108  1.00840009 2.29174683  2.43985827]
        //from 0.1 to 0.7, degree=6
        a1.push_back(-0.03216663533709177);
        b1.push_back(5.359140241417692);
        c1.push_back(-7.308745554603273);
        d1.push_back(3.6604110068015703);
        
        a2.push_back(5.491355198579896);
        b2.push_back(-7.496635998204148);
        c2.push_back(5.619133141454438);
        d2.push_back(-1.761181767348659);
         */
        
        //from 0.1 to 0.7, degree=3
        //a1.push_back(4.17688396);
        //b1.push_back(-13.66592657);
        //c1.push_back(23.74576715);
        //d1.push_back(-14.87368246);
        
        
        //from 0.1 to 0.7, degree=2
        //[[5.91965872,-7.3475699,3.54940138]]
        //a1.push_back(3.54940138);
        //b1.push_back(-7.3475699);
        //c1.push_back(5.91965872);
        
        //from 0.1 to 0.7, degree=1
        //a1.push_back(2.78221164);
        //b1.push_back(-2.61776258);
        
        if(degree==1)
        {
            a1.push_back(2.78221164);
            b1.push_back(-2.61776258);
        }
        else if(degree==2)
        {
            a1.push_back(3.54940138);
            b1.push_back(-7.3475699);
            c1.push_back(5.91965872);
        }
        else if (degree==3)
        {
            /*
            a1.push_back(4.17688396);
            b1.push_back(-13.66592657);
            c1.push_back(23.74576715);
            d1.push_back(-14.87368246);
            */
            //large range = 0.05-1.0
            //[  4.38423127 -13.57853979  19.8459398   -9.81663423]
            a1.push_back(4.38423127);
            b1.push_back(-13.57853979);
            c1.push_back(19.8459398);
            d1.push_back(-9.81663423);
        }
        else if (degree==6)
        {
            /*
            a1.push_back(-0.03216663533709177);
            b1.push_back(5.359140241417692);
            c1.push_back(-7.308745554603273);
            d1.push_back(3.6604110068015703);
            
            a2.push_back(5.491355198579896);
            b2.push_back(-7.496635998204148);
            c2.push_back(5.619133141454438);
            d2.push_back(-1.761181767348659);
            */
            
            //large range = 0.05-1.0
            //[ 1.81885289 -2.860004    5.00751026 -3.55071694 -0.43373061  6.01382611 -8.063566    3.62820803]
            a1.push_back(1.81885289);
            b1.push_back(-2.860004);
            c1.push_back(5.00751026);
            d1.push_back(-3.55071694);
            
            a2.push_back(-0.43373061);
            b2.push_back(6.01382611);
            c2.push_back(-8.063566);
            d2.push_back(3.62820803);
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
    
    
    //encoder.encode(tenth, scale, tenth_plain);
    
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
    
    //vector<int> degrees{1};
    //vector<int> degrees{2};
    vector<int> degrees{3};
    //vector<int> degrees{3,3};
    
    //vector<int> degrees{3,1};
    //vector<int> degrees{3,2};

    
    
    //read data file to get A values
    
    vector<vector<double>> A_message;
    vector<Plaintext> A_plain;
    vector<Ciphertext> A_enc;
    A_enc.reserve(20);
    
    string row;
    //ifstream A_file ("../data/A_values_test_large.txt");
    //ifstream A_file ("../data/A_values_test.txt");
    ifstream A_file ("../big_data/A_values_test.txt");
    string A_values_str;
    int zeroes_to_add;
    int to_rotate = -1; //we need to know how much to rotate B by for concatenation
    
    
    //we can pack many vectors into a single ciphertext
    //each vector takes its size (slot_count) twice to allow for rotation to work
    int max_packed_words = slots / (2 * slot_count);
    //max_packed_words = 40;
    int max_words_packed_in_single_cipher = -1;
    int packed_words = 0;
    
    ///
    
    //max_words_packed_in_single_cipher = 1;
    //max_packed_words = 1;
    
    
    ///
    
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
          //row.push_back(' '); // maybe put back in?
          
          //vector<double> message_temp;
          //Plaintext plain_temp;
          //Ciphertext enc_temp;
          
          size_t pos = 0;
          string A_sub_value;
          while((pos = row.find(' ')) != -1)
          {
              zeroes_to_add--;
              A_sub_value = row.substr(0, pos-1);
              //cout << "trying to doublize: " << A_sub_value << endl;
              double A_sub_value_final = stod(A_sub_value);
              message_temp.push_back(A_sub_value_final);
              row.erase(0, pos+1);
          }
          //cout << "zeroes to add " << zeroes_to_add << endl;
          for(int i = 0; i < zeroes_to_add; i++)
              message_temp.push_back(0.0);
          if(to_rotate == -1)
          {
              to_rotate = -1 * (slot_count - zeroes_to_add);
              //to_rotate = -1 * 1024;
          }
              
          
          int to_replicate = slot_count;//message_temp.size();
          //to_replicate = 1024;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          //for(int i=0;i<to_replicate;i++)
              //message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          //cout << "test: ";
          //for(int p = 0; p < message_temp.size(); p++)
              //cout << message_temp[p] << " ";
          //cout << endl << endl;
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
              //A_plain.push_back(plain_temp);
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
    B_enc.reserve(20);
    
    //ifstream B_file ("../data/B_values_test_large.txt");
    //ifstream B_file ("../data/B_values_test.txt");
    ifstream B_file ("../big_data/B_values_test.txt");
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
          //strip off start and end brackets
          row = row.substr(1,B_values_str.length()-2);
          //row.push_back(' '); //if things break put this back in later
          
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
          //to_replicate = 512;
          //cout << to_replicate << endl;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          
          packed_words++;
          if(packed_words >= max_packed_words)
          {
              //cout << "this many values going in:" << message_temp.size() << endl;
              //cout << "encode ";
              
              encoder.encode(message_temp, scale, plain_temp);
              //cout << "encoded";
              encryptor.encrypt(plain_temp, enc_temp);
              B_enc.push_back(enc_temp);
              //B_plain.push_back(plain_temp);
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
    cout << test_counter << endl;
    
    cout << "concatenate dataset" << endl;
    start = high_resolution_clock::now();
    
    //build encrypted dataset by rotating each query in B, adding together
    //this is effectively concatenation of each row of A and B
    cout << to_rotate << endl;
    vector<Ciphertext> enc_queries;
    enc_queries.reserve(20);
    for(int i = 0; i < A_enc.size(); i++)
    {
        Ciphertext query;
        Ciphertext B_temp = B_enc[i];
        //for(int j = 0; j < 5; j++)
            //evaluator.rotate_vector(B_temp, 1, gal_keys, B_temp);
        evaluator.rotate_vector(B_enc[i], to_rotate, gal_keys, B_temp);
        evaluator.add(A_enc[i], B_temp, query);
        enc_queries.push_back(query);
        
        /*
        //Sanity check on concatenation, packing - APPEARS TO BE WORKING
        Plaintext plain_result1;
        decryptor.decrypt(query, plain_result1);
        //decryptor.decrypt(A_enc[0], plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        int index = 0;
        //cout << A_message[i][3] << endl;
        
        cout << "encrypted: ";
        for(int p = 0; p < result1.size(); p++)
        {
            if(p%512==0)
                cout << endl << "eh" << endl;
            if(result1[p]<0.000001)
                cout << 0 << " ";
            else
                cout << result1[p] << " ";
        }
            
        cout << endl << endl << endl;
        */
        //cout << "message: ";
        //for(int p = 0; p < A_message[i].size(); p++)
            //cout << A_message[0][p] << " ";
        //cout << endl << endl << endl;
        /*
        cout << result1[3] << " =?= " << result1[slot_count+3] << ", truth = " << A_message[i][3] << endl;
        cout << result1[512] << " =?= " << result1[512+1024] << ", truth = " << B_message[i][0] << endl;
        cout << result1[513] << " =?= " << result1[513+1024] << ", truth = " << B_message[i][1] << endl;
        cout << result1[2048] << " =?= " << result1[2048+1024] << ", truth = " << B_message[i][2048] << endl;
        cout << endl;
        */
        
        //cout << "query: ";
        //print_vector(result1, 5, 7);
        //cout << result1.size() << endl;
    }
    
    stop = high_resolution_clock::now();
    cout << endl;
    duration = duration_cast<microseconds>(stop - start);
    cout << "time to concatenate: " << duration.count() / 1000.0 << " milliseconds" << endl;
    
    
    ofstream myfile3;
    myfile3.open ("concatenated_data.txt");
    for(int c = 0; c < enc_queries.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(enc_queries[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            //double mag = 0;
            for(int j = slot_count*2*i; j < slot_count*2*i+1024; j++)
            {
                myfile3 << result[j] << " ";
                //mag+= pow(result[j],2);
            }
            //mag = pow(mag,0.5);
            //cout << mag << endl;
            myfile3 << endl;
        }
    }
    myfile3.close();
    
    /*
    Plaintext plain_result1;
    decryptor.decrypt(enc_queries[0], plain_result1);
    vector<double> result1;
    encoder.decode(plain_result1, result1);
    for(int i = 0; i < max_words_packed_in_single_cipher; i++)
    {
        //double mag = 0;
        for(int j = slot_count*2*i; j < slot_count*2*i+2048; j++)
        {
            cout << result1[j] << " ";
            //mag+= pow(result1[j],2);
        }
        //mag = pow(mag,0.5);
        //cout << mag << "    " << pow(mag,2) << endl;
        cout << endl << endl;
    }*/
    
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
    //ifstream P_file ("../data/features_best_P_value_diagonal_lambda=0.1_margin=0.25_gamma=128_reg=0.1.txt");
    //ifstream P_file ("../data/diagonal_exact_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt");
    //ifstream P_file ("../data/diagonal_approximate_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt");
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
              //cout << P_sub_value_final << " ";
              row.erase(0, pos+1);
          }
          //cout << endl << endl;
          
          int to_replicate = message_temp.size();
          //cout << to_replicate << endl;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[i]);
          
          for(int i=0;i<(max_packed_words-1);i++)
          {
              for(int j=0;j<to_replicate*2;j++)
                  message_temp.push_back(message_temp[j]);
                  //message_temp.push_back(message_temp[i*j]);
          }
          //packed_words++;
          //if(packed_words >= max_packed_words)
          //{
          //cout << message_temp[0] << ", truth = " << 3.052986357943155e-05 << " or " << 0.003704237751662731 << endl;
          
          //cout << "P matrix has this many terms:" << message_temp.size() << endl;
          //cout << message_temp[0] << " " << message_temp[1024] << " " << message_temp[2048] << endl;
          //cout << message_temp[3] << " " << message_temp[1024+3] << " " << message_temp[2048+3] << endl;
          encoder.encode(message_temp, scale, plain_temp);
          //encryptor.encrypt(plain_temp, enc_temp);
          //p_matrix_diagonal.push_back(enc_temp);
          p_matrix_diagonal_plain.push_back(plain_temp);
          p_matrix_diagonal_message.push_back(message_temp);
          
          message_temp.clear();
          //packed_words = 0;
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
    
    cout << "Gamma = " << gamma << endl;
    //cout << "If this isn't 64 I'm gonna scream:" << p_matrix_diagonal_plain.size() << endl;
    /*
    cout <<p_matrix_diagonal_plain.size() << endl;
    
    vector<double> result1;
    encoder.decode(p_matrix_diagonal_plain[0], result1);
    int temp2 = 0;
    for(int i = 0; i < result1.size(); i++)
    {
        cout << result1[i] << " ";
    }
    cout << endl;
     */
    //cout << temp2 << endl;
    
     //print_vector(p_matrix_diagonal_message[0], 32, 7);
    //cout << "p diag plain size: " << p_matrix_diagonal_plain.size() << endl;
    //cout << "done using new diag" << endl;
    
    //p matrix is d_0 by d_1
    //calculate later
    int d_0 = gamma;
    //d_0 = 128;
    int d_1 = 1024;
    //d_1 = 2048;
    int logd_0 = log2(d_0);
    int logd_1 = log2(d_1);
    
    cout << "begin fusion" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    start = high_resolution_clock::now();
    
    vector<Ciphertext> enc_fusions;
    enc_fusions.reserve(20);
    //cout << "here" << endl;
    vector<double> message_zeroes;
    vector<double> message_ones;
    //message_ones.clear();
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
    //cout << "here" << endl;
    encoder.encode(message_zeroes, scale, plain_zeroes);
    encoder.encode(message_ones, scale, plain_ones);
    encoder.encode(message_zeroes2, scale, plain_zeroes2);
    //cout << "encoded" << endl;
    
    for(int c = 0; c < enc_queries.size(); c++)
    //for(int c = 0; c < 4; c++)
    {
        cout << c << " ";
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
            //cout << "D" << i << " " << p_matrix_diagonal_message.size() << endl;
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
        //encrypted_result = encrypted_final_result; ///////////
        for(int j = logd_1-1; j>=logd_0;j--)
        {
            //cout << "rotating by " << pow(2,j) << endl;
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
    
    /*
    Plaintext plain_result1;
    decryptor.decrypt(enc_fusions[0], plain_result1);
    vector<double> result1;
    encoder.decode(plain_result1, result1);
    for(int i=0;i<result1.size(); i++)
    {
        cout << result1[i] << " " << endl;
            
    }
    */
    //Normalization begins here////////////////////////////////////
    cout << "begin normalization" << endl;
    
    
    vector<Ciphertext> normalized_enc_fusions;
    
    start = high_resolution_clock::now();
    for(int i=0; i < enc_fusions.size(); i++)
    //for(int i=0; i < 1; i++)
    {
        Ciphertext cipher = enc_fusions[i];
        
        int vector_size = gamma; // test value
        //vector_size = 128;
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
        
        //begin arbitrary polynomial evaluation
        //cout << "evaluate polynomial" << endl;
        //Calculate approximate inverse norm from squared norm
        //params:
        //coeffs with all the coeffs in order
        Ciphertext result;
        
        Ciphertext x = squared_norm;
        //cout << "starting x current modulus: " << context.get_context_data(x.parms_id())->chain_index() << endl;
        
        /*
        //Squared norm appears to be correct, it's the result of the matrix-vector product that is wrong
        Plaintext plain_result1;
        decryptor.decrypt(squared_norm, plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        cout << "square norm = " << result1[0] << endl;
        */
        
        encryptor.encrypt(plain_zeroes, result);
        int a = 0;//current degree
        int b = 0;//index of degrees
        
        for(int c = 0; c < coeffs.size(); c++)
        {
            //cout << "coeff number: "<<c<<endl;
            Ciphertext inter_result;
            for(int d=0; d<a; d++)
            {
                if(d==0)
                {
                    //cout << "x current modulus: " << context.get_context_data(x.parms_id())->chain_index() << endl;
                    //cout << "d0 mult" << endl;
                    evaluator.mod_switch_to_inplace(coeffs[c], x.parms_id());
                    coeffs[c].scale() = scale;
                    x.scale() = scale;
                    evaluator.multiply_plain(x, coeffs[c], inter_result);
                    evaluator.relinearize_inplace(inter_result, relin_keys);
                    evaluator.rescale_to_next_inplace(inter_result);
                    //cout << "inter result current modulus: " << context.get_context_data(inter_result.parms_id())->chain_index() << endl;
                }
                else
                {
                    //cout << "x current modulus: " << context.get_context_data(x.parms_id())->chain_index() << endl;
                    //cout << "inter x mult" << endl;
                    Ciphertext x_temp = x;
                    evaluator.mod_switch_to_inplace(x_temp, inter_result.parms_id());
                    x_temp.scale() = scale;
                    inter_result.scale() = scale;
                    evaluator.multiply(x_temp, inter_result, inter_result);
                    evaluator.relinearize_inplace(inter_result, relin_keys);
                    evaluator.rescale_to_next_inplace(inter_result);
                    //cout << "inter result current modulus: " << context.get_context_data(inter_result.parms_id())->chain_index() << endl;
                }
            }
            
            //cout << "adding" << endl;
            //cout << context.get_context_data(result.parms_id())->chain_index() << endl;
            //cout << context.get_context_data(inter_result.parms_id())->chain_index() << endl;
            
            //cout << "mod switched" << endl;
            if(a!=0)
            {
                //cout << "inter result current modulus: " << context.get_context_data(inter_result.parms_id())->chain_index() << endl;
                evaluator.mod_switch_to_inplace(result, inter_result.parms_id());
                result.scale() = scale;
                inter_result.scale() = scale;
                evaluator.add_inplace(result, inter_result);
            }
            else
            {
                evaluator.mod_switch_to_inplace(coeffs[c], result.parms_id());
                coeffs[c].scale() = scale;
                result.scale() = scale;
                evaluator.add_plain_inplace(result, coeffs[c]);
            }
            //cout << "result current modulus: " << context.get_context_data(result.parms_id())->chain_index() << endl;
            //cout << "added" << endl;
            a++;
            if(a>degrees[b])
            {
                //cout << endl << "next poly" << endl << endl;
                a=0;
                b++;
                x = result;
                encryptor.encrypt(plain_zeroes, result);
            }
        }
        //end arbitrary polynomial evaluation - this code will be useful later
        
        
        //cout << "done evaluating poly" << endl;
        //now we multiply the original vector by the inverse norm and return that result
        Ciphertext normalized_cipher;
        
        evaluator.mod_switch_to_inplace(cipher, x.parms_id());
        cipher.scale() = x.scale();
        
        /*
        Plaintext plain_result1;
        decryptor.decrypt(x, plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        cout << "inverse norm = " << result1[0] << endl;
        
        decryptor.decrypt(squared_norm, plain_result1);
        encoder.decode(plain_result1, result1);
        cout << "squared norm = " << result1[0] << endl;
        
        decryptor.decrypt(cipher, plain_result1);
        //vector<double> result1;
        encoder.decode(plain_result1, result1);
        double mag = 0;
        for(int j = 0; j < 128; j++)
        {
            //cout << result[j] << " ";
            mag+= pow(result1[j],2);
        }
        mag = pow(mag,0.5);
        cout << "norm = " << mag << endl;
        */
        
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
            for(int j = slot_count*2*i; j < slot_count*2*i+gamma; j++)
            {
                //cout << j << " ";
                
                mag+= pow(result[j],2);
            }
            mag = pow(mag,0.5);
            cout << mag << endl;
            //cout << endl;
        }
    }
    
    cout << endl;
    cout << max_words_packed_in_single_cipher << endl;
    cout << endl;
    
    for(int c = 0; c < enc_fusions.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(enc_fusions[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            double mag = 0;
            //cout << slot_count*2*i << endl;
            //cout << result[slot_count*2*i] << endl;
            for(int j = slot_count*2*i; j < slot_count*2*i+gamma; j++)
            {
                //cout << result[j] << " ";
                mag+= pow(result[j],2);
            }
            mag = pow(mag,0.5);
            cout << mag << "    " << pow(mag,2) << endl;
            //cout << endl << endl;
        }
    }

    
    //string P_file_name_in, string outfile_name, string normalized_outfile_name
    ofstream myfile;
    //myfile.open ("normalized_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_EXACT.txt");
    //myfile.open ("normalized_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY.txt");
    myfile.open (normalized_outfile_name);
    for(int c = 0; c < normalized_enc_fusions.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(normalized_enc_fusions[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            //double mag = 0;
            for(int j = slot_count*2*i; j < slot_count*2*i+gamma; j++)
            {
                myfile << result[j] << " ";
                //mag+= pow(result[j],2);
            }
            //mag = pow(mag,0.5);
            //cout << mag << endl;
            myfile << endl;
        }
    }
    myfile.close();
    
    ofstream myfile2;
    //myfile2.open ("encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_EXACT.txt");
    //myfile2.open ("encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY.txt");
    myfile2.open (outfile_name);
    for(int c = 0; c < enc_fusions.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(enc_fusions[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            //double mag = 0;
            for(int j = slot_count*2*i; j < slot_count*2*i+gamma; j++)
            {
                myfile2 << result[j] << " ";
                //mag+= pow(result[j],2);
            }
            //mag = pow(mag,0.5);
            //cout << mag << endl;
            myfile2 << endl;
        }
    }
    myfile2.close();
    

    

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
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 50, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 50 }));
    
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
    slot_count = 1024;
    cout << "Number of slots: " << slot_count << endl;
    
    //set up constants for normalization
    size_t slots = encoder.slot_count();
    vector<double> linear_weight, linear_bias, constant_approx, neg_half, three_half, half, three_half_times_guess, two;
    Plaintext linear_weight_plain, linear_bias_plain, constant_approx_plain, neg_half_plain, three_half_plain, half_plain, three_half_times_guess_plain, two_plain;
    
    
    //500-4000 train set
    //-6.96535417e+09  7.97134488e+06 -2.49368280e+03  2.78740519e-01
    //2.71838862e-02 -4.26441476e-12
    
    // linear bias, weight over 1-150: [ 0.28190079 -0.00173118]
    //over 36-625 [ 0.10891818 -0.00013434]
    //over 81-1350 [ 7.36341307e-02 -4.18025898e-05]
    for(int i=0; i<slots; i++)
    {
        constant_approx.push_back(0.01);
        
        //over 0.1 to 0.7
        linear_bias.push_back(2.78539063);
        linear_weight.push_back(-2.19074796);
        //constant_approx.push_back(0.025);
        //constant_approx.push_back(6.13117781e-06);
        neg_half.push_back(-0.5);
        three_half.push_back(1.5);
        half.push_back(0.5);
        three_half_times_guess.push_back(0.0375);
        two.push_back(2.0);
    }

    //encode coefficients for normalization now that encoder is defined
    encoder.encode(constant_approx, scale, constant_approx_plain);
    encoder.encode(linear_bias, scale, linear_bias_plain);
    encoder.encode(linear_weight, scale, linear_weight_plain);
    encoder.encode(neg_half, scale, neg_half_plain);
    encoder.encode(three_half, scale, three_half_plain);
    encoder.encode(half, scale, half_plain);
    encoder.encode(three_half_times_guess, scale, three_half_times_guess_plain);
    encoder.encode(two, scale, two_plain);
    
    //read data file to get A values
    
    vector<vector<double>> A_message;
    vector<Plaintext> A_plain;
    vector<Ciphertext> A_enc;
    A_enc.reserve(20);
    
    string row;
    ifstream A_file ("../data/A_values_test.txt");
    string A_values_str;
    int zeroes_to_add;
    int to_rotate = -1; //we need to know how much to rotate B by for concatenation
    
    
    //we can pack many vectors into a single ciphertext
    //each vector takes its size (slot_count) twice to allow for rotation to work
    int max_packed_words = slots / (2 * slot_count);
    //max_packed_words = 40;
    int max_words_packed_in_single_cipher = -1;
    int packed_words = 0;
    
    ///
    
    //max_words_packed_in_single_cipher = 1;
    //max_packed_words = 1;
    
    
    ///
    
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
          //row.push_back(' '); // maybe put back in?
          
          //vector<double> message_temp;
          //Plaintext plain_temp;
          //Ciphertext enc_temp;
          
          size_t pos = 0;
          string A_sub_value;
          while((pos = row.find(' ')) != -1)
          {
              zeroes_to_add--;
              A_sub_value = row.substr(0, pos-1);
              //cout << "trying to doublize: " << A_sub_value << endl;
              double A_sub_value_final = stod(A_sub_value);
              message_temp.push_back(A_sub_value_final);
              row.erase(0, pos+1);
          }
          //cout << "zeroes to add " << zeroes_to_add << endl;
          for(int i = 0; i < zeroes_to_add; i++)
              message_temp.push_back(0.0);
          if(to_rotate == -1)
          {
              to_rotate = -1 * (slot_count - zeroes_to_add);
              //to_rotate = -1 * 1024;
          }
              
          
          int to_replicate = slot_count;//message_temp.size();
          //to_replicate = 1024;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          //for(int i=0;i<to_replicate;i++)
              //message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          //cout << "test: ";
          //for(int p = 0; p < message_temp.size(); p++)
              //cout << message_temp[p] << " ";
          //cout << endl << endl;
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
              //A_plain.push_back(plain_temp);
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
    B_enc.reserve(20);
    
    ifstream B_file ("../data/B_values_test.txt");
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
          //row.push_back(' '); //if things break put this back in later
          
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
          //to_replicate = 512;
          //cout << to_replicate << endl;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[packed_words*slot_count*2+i]);
          
          packed_words++;
          if(packed_words >= max_packed_words)
          {
              //cout << "this many values going in:" << message_temp.size() << endl;
              //cout << "encode ";
              
              encoder.encode(message_temp, scale, plain_temp);
              //cout << "encoded";
              encryptor.encrypt(plain_temp, enc_temp);
              B_enc.push_back(enc_temp);
              //B_plain.push_back(plain_temp);
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
    
    
    
    
    
    
    vector<Ciphertext> p_matrix_diagonal;
    vector<Plaintext> p_matrix_diagonal_plain;
    vector<vector<double>> p_matrix_diagonal_message;
    p_matrix_diagonal.reserve(slot_count);
    
    cout << "building P" << endl;
    //read data file to get best P value
    //string row;
    //ifstream P_file ("../data/features_best_P_value_diagonal_lambda=0.1_margin=0.25_gamma=128_reg=0.1.txt");
    //ifstream P_file ("../data/diagonal_exact_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt");
    
    ifstream P_file ("../data/diagonal_exact_best_P_value_transpose_lambda=0.25_margin=0.25_gamma=64_reg=0.txt");
    string P_values_str;
    message_temp.clear();
    packed_words = 0;
    
    int gamma = 0;
    
    if (P_file.is_open())
    {
      while ( getline (P_file,row) )
      {
          gamma++;
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
              //cout << P_sub_value_final << " ";
              row.erase(0, pos+1);
          }
          //cout << endl << endl;
          
          int to_replicate = message_temp.size();
          //cout << to_replicate << endl;
          for(int i=0;i<to_replicate;i++)
              message_temp.push_back(message_temp[i]);
          
          for(int i=0;i<(max_packed_words-1);i++)
          {
              for(int j=0;j<to_replicate*2;j++)
                  message_temp.push_back(message_temp[j]);
                  //message_temp.push_back(message_temp[i*j]);
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
    //cout << "here" << endl;
    vector<double> message_zeroes;
    vector<double> message_ones;
    //message_ones.clear();
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
    //cout << "here" << endl;
    encoder.encode(message_zeroes, scale, plain_zeroes);
    encoder.encode(message_ones, scale, plain_ones);
    encoder.encode(message_zeroes2, scale, plain_zeroes2);
    //cout << "encoded" << endl;
    
    for(int c = 0; c < enc_queries.size(); c++)
    //for(int c = 0; c < 4; c++)
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
            //cout << "D" << i << " " << p_matrix_diagonal_message.size() << endl;
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
        //encrypted_result = encrypted_final_result; ///////////
        for(int j = logd_1-1; j>=logd_0;j--)
        {
            //cout << "rotating by " << pow(2,j) << endl;
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
        
        int vector_size = gamma; // test value
        
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
        //cout << "set up done" << endl;
        
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
        
        //cout << "prior to first mul" << endl;
        
        
        //cout << "!" << endl;
        
        //linear approximation start
        Ciphertext first_guess;
        //linear_weight_plain, linear_bias_plain
        evaluator.mod_switch_to_inplace(linear_weight_plain, squared_norm.parms_id());
        linear_weight_plain.scale() = squared_norm.scale();
        evaluator.multiply_plain(squared_norm, linear_weight_plain, first_guess);
        evaluator.relinearize_inplace(first_guess, relin_keys);
        evaluator.rescale_to_next_inplace(first_guess);
        
        evaluator.mod_switch_to_inplace(linear_bias_plain, first_guess.parms_id());
        linear_bias_plain.scale() = first_guess.scale();
        evaluator.add_plain_inplace(first_guess, linear_bias_plain);
        //linear approximation end
        //cout << "linear done" << endl;
        
        
        Ciphertext const_approx_plain_temp = first_guess;//constant_approx_plain;
        
        //evaluator.mod_switch_to_inplace(const_approx_plain_temp, squared_norm.parms_id());
        //const_approx_plain_temp.scale() = pow(2,40);
        
        evaluator.mod_switch_to_inplace(squared_norm, const_approx_plain_temp.parms_id());
        squared_norm.scale() = pow(2,40);
        
        evaluator.multiply(squared_norm, const_approx_plain_temp, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        //cout <<"first mul done"<<endl;
        
        
        evaluator.mod_switch_to_inplace(const_approx_plain_temp, squared_norm.parms_id());
        const_approx_plain_temp.scale() = pow(2,40);
        evaluator.multiply(squared_norm, const_approx_plain_temp, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        //cout <<"second mul done"<<endl;
        
        evaluator.mod_switch_to_inplace(neg_half_plain, squared_norm.parms_id());
        neg_half_plain.scale() = pow(2,40);
        evaluator.multiply_plain(squared_norm, neg_half_plain, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        //cout <<"third mul done"<<endl;
        
        evaluator.mod_switch_to_inplace(const_approx_plain_temp, squared_norm.parms_id());
        const_approx_plain_temp.scale() = pow(2,40);
        evaluator.multiply(squared_norm, const_approx_plain_temp, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        //cout <<"foruth mul done"<<endl;
        

        
        evaluator.mod_switch_to_inplace(three_half_times_guess_plain, squared_norm.parms_id());
        three_half_times_guess_plain.scale() = pow(2,40);
        squared_norm.scale() = pow(2,40);
        //cout << temp.scale() << endl;
        evaluator.add_plain(squared_norm, three_half_times_guess_plain, guess);
        //evaluator.relinearize_inplace(guess, relin_keys);
        //evaluator.rescale_to_next_inplace(guess);
        //cout <<"finv end"<<endl;
        //fast inv square root end
        
        
        /*
        Plaintext plain_result1;
        decryptor.decrypt(guess, plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        print_vector(result1, 10, 7);
        */
        
        //goldschmidts begin
        Ciphertext x, h, r;
        int iterations_gold = 4;
        
        
        evaluator.mod_switch_to_inplace(squared_norm_down1, guess.parms_id());
        squared_norm_down1.scale() = pow(2,40);
        evaluator.multiply(squared_norm_down1, guess, x);
        evaluator.relinearize_inplace(x, relin_keys);
        evaluator.rescale_to_next_inplace(x);

        Plaintext half_plain_temp = half_plain;
        
        evaluator.mod_switch_to_inplace(half_plain_temp, guess.parms_id());
        half_plain_temp.scale() = pow(2,40);
        evaluator.multiply_plain(guess, half_plain_temp, h);
        evaluator.relinearize_inplace(h, relin_keys);
        evaluator.rescale_to_next_inplace(h);
        
        for(int i = 0; i < iterations_gold; i++)
        {

            //cout << "time for first g iter" << endl;
            //cout << "goldschmidt iter: " << i << endl;
            
            evaluator.multiply(x, h, temp);
            evaluator.relinearize_inplace(temp, relin_keys);
            evaluator.rescale_to_next_inplace(temp);
            
            evaluator.negate_inplace(temp);
            //cout << "negated" << endl;
            
            //cout <<"!" << endl;
            evaluator.mod_switch_to_inplace(half_plain_temp, temp.parms_id());
            half_plain_temp.scale() = pow(2,40);
            temp.scale() = pow(2,40);
            evaluator.add_plain(temp, half_plain_temp, r);
            //cout << "added" << endl;
            
            //cout <<"!" << endl;
            evaluator.mod_switch_to_inplace(x, r.parms_id());
            x.scale() = pow(2,40);
            evaluator.multiply(x, r, temp);
            evaluator.relinearize_inplace(temp, relin_keys);
            evaluator.rescale_to_next_inplace(temp);
            
            //cout <<"!" << endl;
            evaluator.mod_switch_to_inplace(x, temp.parms_id());
            x.scale() = pow(2,40);
            temp.scale() = pow(2,40);
            evaluator.add(x, temp, x);
            //cout << "line 19 done" << endl;
            
            //cout <<"!" << endl;
            evaluator.mod_switch_to_inplace(h, r.parms_id());
            h.scale() = pow(2,40);
            evaluator.multiply(h, r, temp);
            evaluator.relinearize_inplace(temp, relin_keys);
            evaluator.rescale_to_next_inplace(temp);
            
            //cout <<"!" << endl;
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
        
        /*
        cout << "    + Modulus chain index for cipher: "
             << context.get_context_data(cipher.parms_id())->chain_index() << endl;
        cout << "    + Modulus chain index for guess: "
             << context.get_context_data(guess.parms_id())->chain_index() << endl;
        */
        evaluator.mod_switch_to_inplace(cipher, guess.parms_id());
        cipher.scale() = guess.scale();
        //cout << "made it" << endl;
        evaluator.multiply(cipher, guess, normalized_cipher);
        evaluator.relinearize_inplace(normalized_cipher, relin_keys);
        //cout << "double made it" << endl;
        evaluator.rescale_to_next_inplace(normalized_cipher);
        normalized_enc_fusions.push_back(normalized_cipher);
        //cout << "triple made it" << endl;
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
            for(int j = slot_count*2*i; j < slot_count*2*i+gamma; j++)
            {
                //cout << j << " ";
                
                mag+= pow(result[j],2);
            }
            mag = pow(mag,0.5);
            cout << mag << endl;
            //cout << endl;
        }
    }
    
    cout << endl;
    cout << max_words_packed_in_single_cipher << endl;
    cout << endl;
    
    for(int c = 0; c < enc_fusions.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(enc_fusions[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            double mag = 0;
            //cout << slot_count*2*i << endl;
            //cout << result[slot_count*2*i] << endl;
            for(int j = slot_count*2*i; j < slot_count*2*i+gamma; j++)
            {
                //cout << result[j] << " ";
                mag+= pow(result[j],2);
            }
            mag = pow(mag,0.5);
            cout << mag << "    " << pow(mag,2) << endl;
            //cout << endl << endl;
        }
    }
    
    ofstream myfile;
    myfile.open ("allnewdata_normalized_encrypted_results_goldschmidt_test_lambda=0.25_margin=0.25_gamma=64.txt");
    for(int c = 0; c < normalized_enc_fusions.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(normalized_enc_fusions[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            //double mag = 0;
            for(int j = slot_count*2*i; j < slot_count*2*i+gamma; j++)
            {
                myfile << result[j] << " ";
                //mag+= pow(result[j],2);
            }
            //mag = pow(mag,0.5);
            //cout << mag << endl;
            myfile << endl;
        }
    }
    myfile.close();
    
    ofstream myfile2;
    myfile2.open ("allnewdata_encrypted_results_goldschmidt_test_lambda=0.25_margin=0.25_gamma=64.txt");
    for(int c = 0; c < enc_fusions.size(); c++)
    {
        Plaintext plain_result;
        decryptor.decrypt(enc_fusions[c], plain_result);
        vector<double> result;
        encoder.decode(plain_result, result);
        for(int i = 0; i < max_words_packed_in_single_cipher; i++)
        {
            //double mag = 0;
            for(int j = slot_count*2*i; j < slot_count*2*i+gamma; j++)
            {
                myfile2 << result[j] << " ";
                //mag+= pow(result[j],2);
            }
            //mag = pow(mag,0.5);
            //cout << mag << endl;
            myfile2 << endl;
        }
    }
    myfile2.close();

}

void encrypted_feature_fusion_SIMD_polynomial()
{
    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    //size_t poly_modulus_degree = 8192;//16384;
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 60 }));
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
    size_t slots = encoder.slot_count();
    
    
    
    //set up coefficients for normalization
    //size_t slots = encoder.slot_count();
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
    }

    encoder.encode(a1, scale, a1_plain);
    encoder.encode(b1, scale, b1_plain);
    encoder.encode(c1, scale, c1_plain);
    encoder.encode(d1, scale, d1_plain);
    encoder.encode(a2, scale, a2_plain);
    encoder.encode(b2, scale, b2_plain);
    encoder.encode(c2, scale, c2_plain);
    
    
    vector<double> query_message;
    Plaintext query_plain;
    Ciphertext query_enc;
    vector<Ciphertext> enc_queries;
    
    
    string row;
    //ifstream A_file ("../../data/A_values_transpose.txt");
    ifstream A_file ("../data/A_values_transpose.txt");
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
          row = row.substr(1,row.length()-2);
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
          cout << "A in" << endl;
          encoder.encode(message_temp, scale, plain_temp);
          encryptor.encrypt(plain_temp, enc_temp);
          enc_queries.push_back(enc_temp);
          message_temp.clear();
      }
    A_file.close();
    }
    
    //ifstream B_file ("../../data/B_values_transpose.txt");
    ifstream B_file ("../data/B_values_transpose.txt");
    string B_values_str;
    
    message_temp.clear();
    //vector<double> message_temp;
    //Plaintext plain_temp;
    //Ciphertext enc_temp;
    
    if (B_file.is_open())
    {
      cout << "build B" << endl;
      while ( getline (B_file,row) )
      {
          //remove commas, source: https://stackoverflow.com/questions/20326356/how-to-remove-all-the-occurrences-of-a-char-in-c-string
          row.erase(remove(row.begin(), row.end(), ','), row.end());
          
          //zeroes_to_add = slot_count;
          //strip off start and end brackets
          row = row.substr(1,row.length()-2);
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
          cout << "B in" << endl;
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
    //vector<string> filenames{"../../data/best_P_value_transpose_lambda=0.5_margin=0.5.txt"};
    vector<string> filenames{"../data/best_P_value_transpose_lambda=0.5_margin=0.5.txt"};
    
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
        
        
        int index_of_filenames = q;
        ifstream P_file (filenames[index_of_filenames]);
        
        string P_values_str;
        //string row;
        int packed_words = 0;
        int max_packed_words = 1;
        int max_dim = 128;
        if (P_file.is_open())
        {
          cout << "building P" << endl;
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
                  //cout << P_sub_value << endl;
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
        int d_1 = 5;//in dimension
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
        cout << enc_queries.size() << " " << d_1 << " " << p_matrix_plain.size() << endl;
        for(int i=0; i<d_0; i++)
        {
            for(int j=0; j<d_1; j++)
            {
                //cout << j << endl;
                //if(j>=enc_queries.size())
                //{
                    //cout << "breaking" << endl;
                    //break;
                //}
                   
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
        
        /*
        Plaintext plain_result1;
        decryptor.decrypt(enc_fusions[0], plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        print_vector(result1, 3, 7);
        */
        
        
        //Normalization begins here////////////////////////////////////
        cout << "begin normalization" << endl;
        
        
        vector<Ciphertext> normalized_enc_fusions;
        
        start = high_resolution_clock::now();
        
        Ciphertext squared_norm, cipher;
        
        evaluator.square(enc_fusions[0], squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        
        for(int i=1; i < enc_fusions.size(); i++)
        {
            evaluator.square(enc_fusions[i], cipher);
            evaluator.relinearize_inplace(cipher, relin_keys);
            evaluator.rescale_to_next_inplace(cipher);
            
            evaluator.add_inplace(squared_norm, cipher);
            
            
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

        //poly approx begins
        
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
        
        //end poly approx
        
        //goldschmidts end
        //guess is our inverse norm
        
        
        /*
        //now we multiply the original vector by the inverse norm and return that result
        Ciphertext normalized_cipher;
        
        evaluator.mod_switch_to_inplace(cipher, guess.parms_id());
        cipher.scale() = guess.scale();
        
        evaluator.multiply(cipher, guess, normalized_cipher);
        evaluator.relinearize_inplace(normalized_cipher, relin_keys);
        evaluator.rescale_to_next_inplace(normalized_cipher);
        normalized_enc_fusions.push_back(normalized_cipher);
         */
        //mult by all
        
        for(int i=0; i < enc_fusions.size(); i++)
        {
            Ciphertext normalized_cipher;
            
            evaluator.mod_switch_to_inplace(enc_fusions[i], result.parms_id());
            enc_fusions[i].scale() = result.scale();
            
            evaluator.multiply(enc_fusions[i], result, normalized_cipher);
            evaluator.relinearize_inplace(normalized_cipher, relin_keys);
            evaluator.rescale_to_next_inplace(normalized_cipher);
            normalized_enc_fusions.push_back(normalized_cipher);
        }
        
        
        stop = high_resolution_clock::now();
        cout << endl;
        duration = duration_cast<microseconds>(stop - start);
        cout << "time to normalize: " << duration.count() / 1000.0 << " milliseconds" << endl;
        cout << "normalization complete" << endl;
        
        //cout << "num matvec mult ciphers: " << enc_fusions.size() << endl;
        //cout << "num resultant ciphers: " << normalized_enc_fusions.size() << endl;
        
        
        for(int c = 0; c < enc_fusions.size(); c++)
        {
            Plaintext plain_result;
            decryptor.decrypt(enc_fusions[c], plain_result);
            vector<double> result;
            encoder.decode(plain_result, result);
            for(int i = 0; i < 40; i++)
            {
                
                //double mag = 0;
                //for(int j = slot_count*2*i; j < slot_count*2*i+2; j++)
                //{
                    cout << result[i];
                    //if(j != slot_count*2*i+2-1)
                        cout << " ";
                    //mag+= pow(result[j],2);
                //}
                //mag = pow(mag,0.5);
                //cout << mag << endl;
                //cout << ";" << labels[i*(c+1)] << endl;
            
            }
            cout << endl << endl;
        }
        
        
        for(int c = 0; c < normalized_enc_fusions.size(); c++)
        {
            Plaintext plain_result;
            decryptor.decrypt(normalized_enc_fusions[c], plain_result);
            vector<double> result;
            encoder.decode(plain_result, result);
            for(int i = 0; i < 40; i++)
            {
                
                //double mag = 0;
                //for(int j = slot_count*2*i; j < slot_count*2*i+2; j++)
                //{
                    cout << result[i];
                    //if(j != slot_count*2*i+2-1)
                        cout << " ";
                    //mag+= pow(result[j],2);
                //}
                //mag = pow(mag,0.5);
                //cout << mag << endl;
                //cout << ";" << labels[i*(c+1)] << endl;
            
            }
            cout << endl << endl;
        }
        cout << endl;
    }
}


void encrypted_feature_fusion_SIMD_goldschmidt()
{
    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    //size_t poly_modulus_degree = 8192;//16384;
    size_t poly_modulus_degree = 32768;//16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    //parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 60 }));
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
    size_t slots = encoder.slot_count();
    
    
    
    //set up constants for normalization
    //size_t slots = encoder.slot_count();
    vector<double> constant_approx, neg_half, three_half, half, three_half_times_guess, two;
    Plaintext constant_approx_plain, neg_half_plain, three_half_plain, half_plain, three_half_times_guess_plain, two_plain;
    
    
    for(int i=0; i<slots; i++)
    {
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
    
    
    vector<double> query_message;
    Plaintext query_plain;
    Ciphertext query_enc;
    vector<Ciphertext> enc_queries;
    
    
    string row;
    //ifstream A_file ("../../data/A_values_transpose.txt");
    ifstream A_file ("../data/A_values_transpose.txt");
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
          row = row.substr(1,row.length()-2);
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
          cout << "A in" << endl;
          encoder.encode(message_temp, scale, plain_temp);
          encryptor.encrypt(plain_temp, enc_temp);
          enc_queries.push_back(enc_temp);
          message_temp.clear();
      }
    A_file.close();
    }
    
    //ifstream B_file ("../../data/B_values_transpose.txt");
    ifstream B_file ("../data/B_values_transpose.txt");
    string B_values_str;
    
    message_temp.clear();
    //vector<double> message_temp;
    //Plaintext plain_temp;
    //Ciphertext enc_temp;
    
    if (B_file.is_open())
    {
      cout << "build B" << endl;
      while ( getline (B_file,row) )
      {
          //remove commas, source: https://stackoverflow.com/questions/20326356/how-to-remove-all-the-occurrences-of-a-char-in-c-string
          row.erase(remove(row.begin(), row.end(), ','), row.end());
          
          //zeroes_to_add = slot_count;
          //strip off start and end brackets
          row = row.substr(1,row.length()-2);
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
          cout << "B in" << endl;
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
    //vector<string> filenames{"../../data/best_P_value_transpose_lambda=0.5_margin=0.5.txt"};
    vector<string> filenames{"../data/best_P_value_transpose_lambda=0.5_margin=0.5.txt"};
    
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
        
        
        int index_of_filenames = q;
        ifstream P_file (filenames[index_of_filenames]);
        
        string P_values_str;
        //string row;
        int packed_words = 0;
        int max_packed_words = 1;
        int max_dim = 128;
        if (P_file.is_open())
        {
          cout << "building P" << endl;
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
                  //cout << P_sub_value << endl;
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
        int d_1 = 5;//in dimension
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
        cout << enc_queries.size() << " " << d_1 << " " << p_matrix_plain.size() << endl;
        for(int i=0; i<d_0; i++)
        {
            for(int j=0; j<d_1; j++)
            {
                //cout << j << endl;
                //if(j>=enc_queries.size())
                //{
                    //cout << "breaking" << endl;
                    //break;
                //}
                   
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
        
        /*
        Plaintext plain_result1;
        decryptor.decrypt(enc_fusions[0], plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        print_vector(result1, 3, 7);
        */
        
        
        //Normalization begins here////////////////////////////////////
        cout << "begin normalization" << endl;
        
        
        vector<Ciphertext> normalized_enc_fusions;
        
        start = high_resolution_clock::now();
        
        Ciphertext squared_norm, cipher;
        
        evaluator.square(enc_fusions[0], squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);
        
        for(int i=1; i < enc_fusions.size(); i++)
        {
            evaluator.square(enc_fusions[i], cipher);
            evaluator.relinearize_inplace(cipher, relin_keys);
            evaluator.rescale_to_next_inplace(cipher);
            
            evaluator.add_inplace(squared_norm, cipher);
            
            
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


        //fast inverse square root, one iteration
        
        Ciphertext guess, temp;
        
        evaluator.mod_switch_to_inplace(constant_approx_plain, squared_norm.parms_id());
        constant_approx_plain.scale() = pow(2,40);
        evaluator.multiply_plain(squared_norm, constant_approx_plain, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);

        evaluator.mod_switch_to_inplace(constant_approx_plain, squared_norm.parms_id());
        constant_approx_plain.scale() = pow(2,40);
        evaluator.multiply_plain(squared_norm, constant_approx_plain, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);

        
        evaluator.mod_switch_to_inplace(neg_half_plain, squared_norm.parms_id());
        neg_half_plain.scale() = pow(2,40);
        evaluator.multiply_plain(squared_norm, neg_half_plain, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);

        
        evaluator.mod_switch_to_inplace(constant_approx_plain, squared_norm.parms_id());
        constant_approx_plain.scale() = pow(2,40);
        evaluator.multiply_plain(squared_norm, constant_approx_plain, squared_norm);
        evaluator.relinearize_inplace(squared_norm, relin_keys);
        evaluator.rescale_to_next_inplace(squared_norm);

        
        evaluator.mod_switch_to_inplace(three_half_times_guess_plain, squared_norm.parms_id());
        three_half_times_guess_plain.scale() = pow(2,40);
        squared_norm.scale() = pow(2,40);

        evaluator.add_plain(squared_norm, three_half_times_guess_plain, guess);

        
        Plaintext plain_result1;
        decryptor.decrypt(guess, plain_result1);
        vector<double> result1;
        encoder.decode(plain_result1, result1);
        //print_vector(result1, 10, 7);
        
        
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
            
            evaluator.multiply(x, h, temp);
            evaluator.relinearize_inplace(temp, relin_keys);
            evaluator.rescale_to_next_inplace(temp);
            
            evaluator.negate_inplace(temp);

            evaluator.mod_switch_to_inplace(half_plain, temp.parms_id());
            half_plain.scale() = pow(2,40);
            temp.scale() = pow(2,40);
            evaluator.add_plain(temp, half_plain, r);

            evaluator.mod_switch_to_inplace(x, r.parms_id());
            x.scale() = pow(2,40);
            evaluator.multiply(x, r, temp);
            evaluator.relinearize_inplace(temp, relin_keys);
            evaluator.rescale_to_next_inplace(temp);
;
            evaluator.mod_switch_to_inplace(x, temp.parms_id());
            x.scale() = pow(2,40);
            temp.scale() = pow(2,40);
            evaluator.add(x, temp, x);

            evaluator.mod_switch_to_inplace(h, r.parms_id());
            h.scale() = pow(2,40);
            evaluator.multiply(h, r, temp);
            evaluator.relinearize_inplace(temp, relin_keys);
            evaluator.rescale_to_next_inplace(temp);

            evaluator.mod_switch_to_inplace(h, temp.parms_id());
            h.scale() = pow(2,40);
            temp.scale() = pow(2,40);
            evaluator.add(h, temp, h);
        }
        
        
        
        evaluator.add(h, h, guess);
        //goldschmidts end
        //guess is our inverse norm
        
        
        /*
        //now we multiply the original vector by the inverse norm and return that result
        Ciphertext normalized_cipher;
        
        evaluator.mod_switch_to_inplace(cipher, guess.parms_id());
        cipher.scale() = guess.scale();
        
        evaluator.multiply(cipher, guess, normalized_cipher);
        evaluator.relinearize_inplace(normalized_cipher, relin_keys);
        evaluator.rescale_to_next_inplace(normalized_cipher);
        normalized_enc_fusions.push_back(normalized_cipher);
         */
        //mult by all
        
        for(int i=0; i < enc_fusions.size(); i++)
        {
            Ciphertext normalized_cipher;
            
            evaluator.mod_switch_to_inplace(enc_fusions[i], guess.parms_id());
            enc_fusions[i].scale() = guess.scale();
            
            evaluator.multiply(enc_fusions[i], guess, normalized_cipher);
            evaluator.relinearize_inplace(normalized_cipher, relin_keys);
            evaluator.rescale_to_next_inplace(normalized_cipher);
            normalized_enc_fusions.push_back(normalized_cipher);
        }
        
        
        stop = high_resolution_clock::now();
        cout << endl;
        duration = duration_cast<microseconds>(stop - start);
        cout << "time to normalize: " << duration.count() / 1000.0 << " milliseconds" << endl;
        cout << "normalization complete" << endl;
        
        //cout << "num matvec mult ciphers: " << enc_fusions.size() << endl;
        //cout << "num resultant ciphers: " << normalized_enc_fusions.size() << endl;
        
        
        for(int c = 0; c < enc_fusions.size(); c++)
        {
            Plaintext plain_result;
            decryptor.decrypt(enc_fusions[c], plain_result);
            vector<double> result;
            encoder.decode(plain_result, result);
            for(int i = 0; i < 40; i++)
            {
                
                //double mag = 0;
                //for(int j = slot_count*2*i; j < slot_count*2*i+2; j++)
                //{
                    cout << result[i];
                    //if(j != slot_count*2*i+2-1)
                        cout << " ";
                    //mag+= pow(result[j],2);
                //}
                //mag = pow(mag,0.5);
                //cout << mag << endl;
                //cout << ";" << labels[i*(c+1)] << endl;
            
            }
            cout << endl << endl;
        }
        
        
        for(int c = 0; c < normalized_enc_fusions.size(); c++)
        {
            Plaintext plain_result;
            decryptor.decrypt(normalized_enc_fusions[c], plain_result);
            vector<double> result;
            encoder.decode(plain_result, result);
            for(int i = 0; i < 40; i++)
            {
                
                //double mag = 0;
                //for(int j = slot_count*2*i; j < slot_count*2*i+2; j++)
                //{
                    cout << result[i];
                    //if(j != slot_count*2*i+2-1)
                        cout << " ";
                    //mag+= pow(result[j],2);
                //}
                //mag = pow(mag,0.5);
                //cout << mag << endl;
                //cout << ";" << labels[i*(c+1)] << endl;
            
            }
            cout << endl << endl;
        }
        cout << endl;
    }
}


void Matrix_Vector_Multiplication_Hybrid_Test()
{

    cout << "TEST HYBRID" << endl;
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
    //slot_count = 8;
    size_t slots = encoder.slot_count();
    cout << "Number of slots: " << slots << endl;
    
    
    
    
    //read data file to get A values
    
    vector<double> query_message;
    Plaintext query_plain;
    Ciphertext query_enc;
    
    int delta = 1024;
    int gamma = 512;
    gamma = 64;
    
    for(int i=0; i<delta; i++)
    {
        query_message.push_back(0.123);
    }
    encoder.encode(query_message, scale, query_plain);
    encryptor.encrypt(query_plain, query_enc);
    
    /*
    vector<string> filenames{"../diagonal_random_P_value_transpose_indim=128_outdim=1.txt",
        "../diagonal_random_P_value_transpose_indim=128_outdim=2.txt","../diagonal_random_P_value_transpose_indim=128_outdim=4.txt",
        "../diagonal_random_P_value_transpose_indim=128_outdim=8.txt","../diagonal_random_P_value_transpose_indim=128_outdim=16.txt",
        "../diagonal_random_P_value_transpose_indim=128_outdim=32.txt","../diagonal_random_P_value_transpose_indim=128_outdim=64.txt"
    };*/
    vector<string> filenames{"../diagonal_random_P_value_transpose_indim=1024_outdim=64.txt"};
    
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
        //cout << pow(2,q) << endl;
        vector<double> message_temp;
        Plaintext plain_temp;
        Ciphertext enc_temp;
        
        
        vector<Ciphertext> p_matrix_diagonal;
        vector<Plaintext> p_matrix_diagonal_plain;
        vector<vector<double>> p_matrix_diagonal_message;
        p_matrix_diagonal.reserve(slot_count);
        
        
        /*
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
        */
        
        for(int i = 0; i < gamma; i++)
        {
            for(int j = 0; j < delta; j++)
                message_temp.push_back(0.123);
            encoder.encode(message_temp, scale, plain_temp);
            //encryptor.encrypt(plain_temp, enc_temp);
            //p_matrix_diagonal.push_back(enc_temp);
            p_matrix_diagonal_plain.push_back(plain_temp);
            p_matrix_diagonal_message.push_back(message_temp);
            message_temp.clear();
            //cout << "here" << endl;
        }

        //p matrix is d_0 by d_1
        //calculate later
        //int d_0 = pow(2,q);
        //int d_1 = 128;
        //int logd_0 = q;
        //int logd_1 = 7;
        
        int d_0 = gamma;
        int d_1 = delta;
        int logd_0 = log2(gamma);
        int logd_1 = log2(delta);
        
        
        cout << "begin fusion" << endl;
        //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
        
        
        vector<Ciphertext> enc_fusions;
        Ciphertext enc_fusion;
        
        
        vector<long> times;
        vector<int> ns {1,100,1000};
        
        for(int c = 0; c < ns.size(); c++)
        {
            int n = ns[c];
            int n_prime = ceil(n/floor(slots/(2*delta)));
            cout << "n: " << n << ", n prime: " << n_prime << endl;
            
            start = high_resolution_clock::now();
            for(int d = 0; d < n_prime; d++)
            {
                
                //Ciphertext encrypted_final_result = hybrid_matmul(p_matrix_diagonal_plain, p_matrix_diagonal_message, query_enc, d_0, d_1, &encryptor, &evaluator, relin_keys, gal_keys, plain_zeroes, plain_ones);
                
                //int logd_0 = log2(d_0);
                //int logd_1 = log2(d_1);
                Ciphertext encrypted_result;
                //HERE
                encryptor.encrypt(plain_zeroes, encrypted_result);
                
                evaluator.multiply_plain(encrypted_result, plain_ones, encrypted_result);
                evaluator.relinearize_inplace(encrypted_result, relin_keys);
                evaluator.rescale_to_next_inplace(encrypted_result); //level of encrypted_result must match sub result, which will be rescaled once

                Ciphertext encrypted_sub_result;

                
                for(int i = 0; i < d_0; i++)
                {
                    //cout << "iteration " << i << " of mat-vec mult" << endl;;
                    if(!zero_vector(p_matrix_diagonal_message[i])) //cannot plainmult a vector of all zeroes
                    {
                        //cout << "size: " << p_matrix_diagonal_plain.size() << endl;
                        //cout << "d_0: " << d_0 << endl << endl;
                        
                        evaluator.multiply_plain(query_enc, p_matrix_diagonal_plain[i], encrypted_sub_result);
                        evaluator.relinearize_inplace(encrypted_sub_result, relin_keys);
                        evaluator.rescale_to_next_inplace(encrypted_sub_result);
                        evaluator.add_inplace(encrypted_result, encrypted_sub_result);
                    }
                    evaluator.rotate_vector(query_enc, 1, gal_keys, query_enc);
                }

                Ciphertext encrypted_final_result = encrypted_result;

                
                
                for(int j = logd_1-1; j>=logd_0;j--)
                {
                    evaluator.rotate_vector(encrypted_result, pow(2,j), gal_keys, encrypted_result);
                    Ciphertext temp;
                    evaluator.add(encrypted_final_result, encrypted_result, temp);
                    encrypted_result = temp;
                    encrypted_final_result = temp;
                }
                
                
            }
            stop = high_resolution_clock::now();
            duration = duration_cast<microseconds>(stop - start);
            times.push_back(duration.count() / 1000.0);
            cout << "n=" << n << ", time to fuse: " << duration.count() / 1000.0 << " milliseconds" << endl;
        }
        
        //enc_fusions.push_back(encrypted_final_result);

        //cout << times[0] << endl;
        
        
        //Plaintext plain_result1;
        //decryptor.decrypt(enc_fusions[0], plain_result1);
        //vector<double> result1;
        //encoder.decode(plain_result1, result1);
        //print_vector(result1, 3, 7);
        
        cout << endl;
    }
}


void Matrix_Vector_Multiplication_SIMD_Test()
{
    cout << "TEST SIMD" << endl;
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
    //slot_count = 8;
    cout << "Number of slots: " << slot_count << endl;
    size_t slots = encoder.slot_count();
    
    
    int gamma = 512;
    gamma = 64;
    int delta = 1024;
    
    
    
    vector<double> query_message;
    Plaintext query_plain;
    Ciphertext query_enc;
    vector<Ciphertext> enc_queries;
    
    for(int i=0; i<slots; i++)
    {
        query_message.push_back(0.123);
    }
    for(int i=0; i < delta; i++)
    {
        encoder.encode(query_message, scale, query_plain);
        encryptor.encrypt(query_plain, query_enc);
        enc_queries.push_back(query_enc);
    }
    
    
    /*
    vector<string> filenames{"../data/random_P_value_transpose_indim=2_outdim=1.txt",
        "../data/random_P_value_transpose_indim=4_outdim=1.txt","../data/random_P_value_transpose_indim=8_outdim=1.txt",
        "../data/random_P_value_transpose_indim=16_outdim=1.txt","../data/random_P_value_transpose_indim=32_outdim=1.txt",
        "../data/random_P_value_transpose_indim=64_outdim=1.txt","../data/random_P_value_transpose_indim=128_outdim=1.txt"
    };*/
    vector<string> filenames{"../diagonal_random_P_value_transpose_indim=1024_outdim=64.txt"};
    
    
    
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
        
        /*
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
          }
        P_file.close();
        }
         */
        
        for(int j = 0; j < slots; j++)
            message_temp.push_back(0.123);
        
        //for(int i = 0; i < gamma * delta; i++)
        for(int i = 0; i < 1; i++)
        {
            
            
            encoder.encode(message_temp, scale, plain_temp);
            //encryptor.encrypt(plain_temp, enc_temp);
            //p_matrix.push_back(enc_temp);
            p_matrix_plain.push_back(plain_temp);
            p_matrix_message.push_back(message_temp);
            
            
        }
        
        //p matrix is d_0 by d_1
        //calculate later
        int d_0 = gamma;
        int d_1 = delta;
        int logd_0 = log2(d_0);
        int logd_1 = log2(d_1);
        
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
                //if(!zero_vector(p_matrix_message[j+i*d_1])) //cannot plainmult a vector of all zeroes // i used to be the index
                if(!zero_vector(p_matrix_message[0])) //cannot plainmult a vector of all zeroes
                {
                    //cout << "size: " << p_matrix_diagonal_plain.size() << endl;
                    //cout << "d_0: " << d_0 << endl << endl;
                    
                    evaluator.multiply_plain(query_enc, p_matrix_plain[0], encrypted_sub_result);
                    //evaluator.multiply_plain(query_enc, p_matrix_plain[j+i*d_1], encrypted_sub_result);
                    evaluator.relinearize_inplace(encrypted_sub_result, relin_keys);
                    evaluator.rescale_to_next_inplace(encrypted_sub_result);
                    if(j!=0)
                        evaluator.add_inplace(encrypted_result, encrypted_sub_result);
                    else
                        encrypted_result = encrypted_sub_result;
                }
                
            }
            //enc_fusions.push_back(encrypted_result);
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

void Matrix_Vector_Multiplication_Naive_Test()
{

    cout << "TEST NAIVE" << endl;
    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 8192;//16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
    
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
    //slot_count = 100;
    cout << "Number of slots: " << slot_count << endl;
    size_t slots = encoder.slot_count();
    
    int delta = 1024;
    int gamma = 512;
    gamma = 64;
    
    vector<Ciphertext> enc_queries;
    enc_queries.reserve(1000);
    
    vector<double> query_message;
    Plaintext query_plain;
    Ciphertext query_enc;
    for(int j=0; j<1000;j++)
    {
        for(int i=0; i<slots; i++)
        {
            //query_message.push_back(rand());
            query_message.push_back(0.123);
        }
        encoder.encode(query_message, scale, query_plain);
        encryptor.encrypt(query_plain, query_enc);
        enc_queries.push_back(query_enc);
        //cout << "pushed" << endl;
        query_message.clear();
    }
    cout << "queries built" << endl;
    vector<int> num_queries{1,10,20,30,40,50,60,70,80,90,100};
    
    
    //vector<string> filenames{"best_P_value_transpose_lambda=0.5_margin=0.5_rows.txt"
    //};
    
    
    /*
    vector<string> filenames{"../data/random_P_value_transpose_indim=2_outdim=1.txt",
        "../data/random_P_value_transpose_indim=4_outdim=1.txt","../data/random_P_value_transpose_indim=8_outdim=1.txt",
        "../data/random_P_value_transpose_indim=16_outdim=1.txt","../data/random_P_value_transpose_indim=32_outdim=1.txt",
        "../data/random_P_value_transpose_indim=64_outdim=1.txt","../data/random_P_value_transpose_indim=128_outdim=1.txt"
    };*/
    
    vector<string> filenames{"../diagonal_random_P_value_transpose_indim=1024_outdim=64.txt"};
    
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
    //vector<vector<double>> time;
    //for(int index_of_filenames = 0; index_of_filenames<filenames.size();index_of_filenames++)
    //{
        //cout << "Outer iteration: " << filenames[index_of_filenames] << endl;
        //vector<double> times;
        //for(int q = 0; q < 1; q++)
        //{
            //cout << q << endl;
            vector<double> message_temp;
            Plaintext plain_temp;
            Ciphertext enc_temp;
            
            
            vector<Ciphertext> p_matrix_diagonal;
            vector<Plaintext> p_matrix_diagonal_plain;
            vector<vector<double>> p_matrix_diagonal_message;
            p_matrix_diagonal.reserve(slot_count);
            
            /*
            cout << "building P" << endl;
            //int index_of_filenames = q;
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
                      p_matrix_diagonal.push_back(enc_temp);
                      p_matrix_diagonal_plain.push_back(plain_temp);
                      p_matrix_diagonal_message.push_back(message_temp);
                      
                      message_temp.clear();
                  }
              }
            P_file.close();
            }
            */
            //p matrix is d_0 by d_1
            //calculate later
            //int d_0 = 8;
            //int d_1 = p_matrix_diagonal_plain.size();
            //int logd_0 = 3;
            //int logd_1 = 1;
            
            
            for(int i = 0; i < gamma; i++)
            {
                for(int j = 0; j < delta; j++)
                    message_temp.push_back(0.123);
                encoder.encode(message_temp, scale, plain_temp);
                //encryptor.encrypt(plain_temp, enc_temp);
                //p_matrix_diagonal.push_back(enc_temp);
                p_matrix_diagonal_plain.push_back(plain_temp);
                p_matrix_diagonal_message.push_back(message_temp);
                message_temp.clear();
                //cout << "here" << endl;
            }

            //p matrix is d_0 by d_1
            //calculate later
            //int d_0 = pow(2,q);
            //int d_1 = 128;
            //int logd_0 = q;
            //int logd_1 = 7;
            
            int d_0 = gamma;
            int d_1 = delta;
            int logd_0 = log2(gamma);
            int logd_1 = log2(delta);
            
            
            
            //////////////
            
            cout << "begin fusion" << endl;
            //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
            
            
            vector<Ciphertext> enc_fusions;
            Ciphertext enc_fusion;
            
            
            vector<long> times;
            vector<int> ns {1,100,1000};
            
            for(int c = 0; c < ns.size(); c++)
            {
                int n = ns[c];
                int n_prime = ceil(n/floor(slots/(2*delta)));
                cout << "n: " << n << ", n prime: " << n_prime << endl;
                
                start = high_resolution_clock::now();
                //for(int d = 0; d < n_prime; d++)
                //{

                    vector<Ciphertext> enc_fusions;
                    Ciphertext enc_fusion;
                    vector<Ciphertext> temp_enc_fusions;
                    
                    vector<Plaintext> masks;
                    Plaintext mask_temp_plain;
                    vector<double> mask_temp_message;
                    for(int i=0; i<d_0; i++)
                    {
                        for(int j=0; j<d_1; j++)
                        {
                            if(i!=j)
                                mask_temp_message.push_back(0.0);
                            else
                                mask_temp_message.push_back(1.0);
                        }
                        encoder.encode(mask_temp_message,scale,mask_temp_plain);
                        mask_temp_message.clear();
                        masks.push_back(mask_temp_plain);
                    }
                    
                    
                    Ciphertext encrypted_result;
                    for(int i=0; i<n_prime;i++)
                    {
                        
                        for(int j=0; j<d_0; j++)
                        {
                            //cout << j<< endl;
                            evaluator.multiply_plain(enc_queries[i], p_matrix_diagonal_plain[j], encrypted_result);
                            evaluator.relinearize_inplace(encrypted_result, relin_keys);
                            evaluator.rescale_to_next_inplace(encrypted_result);
                            
                            
                            for(int k = 0; k < logd_1; k++)
                            {
                                Ciphertext temp;
                                evaluator.rotate_vector(encrypted_result, pow(2, k), gal_keys, temp);
                                evaluator.add(encrypted_result, temp, encrypted_result);
                            }
                            
                            //cout << "before mod switch" << endl;
                            evaluator.mod_switch_to_inplace(masks[j], encrypted_result.parms_id());
                            masks[j].scale() = pow(2,40);
                            //cout << "after mod switch" << endl;
                            
                            evaluator.multiply_plain(encrypted_result, masks[j], encrypted_result);
                            evaluator.relinearize_inplace(encrypted_result, relin_keys);
                            evaluator.rescale_to_next_inplace(encrypted_result);
                            
                            temp_enc_fusions.push_back(encrypted_result);
                        }
                        Ciphertext result;
                        //cout << "put it all together" << endl;
                        for(int j=0; j< temp_enc_fusions.size(); j++)
                        {
                            
                            if(j==0)
                                result = temp_enc_fusions[j];
                            else
                            {
                                evaluator.add_inplace(result, temp_enc_fusions[j]);
                            }
                        }
                        //cout << "done" << endl;
                        //enc_fusions.push_back(result);
                    
                    
                    }
                stop = high_resolution_clock::now();
                duration = duration_cast<microseconds>(stop - start);
                times.push_back(duration.count() / 1000.0);
                cout << "n=" << n << ", time to fuse: " << duration.count() / 1000.0 << " milliseconds" << endl;
            }
            
            ///////////////
            
            /*
            cout << "begin fusion" << endl;
            //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
            start = high_resolution_clock::now();
            
            vector<Ciphertext> enc_fusions;
            Ciphertext enc_fusion;
            vector<Ciphertext> temp_enc_fusions;
            
            vector<Plaintext> masks;
            Plaintext mask_temp_plain;
            vector<double> mask_temp_message;
            for(int i=0; i<d_0; i++)
            {
                for(int j=0; j<d_1; j++)
                {
                    if(i!=j)
                        mask_temp_message.push_back(0.0);
                    else
                        mask_temp_message.push_back(1.0);
                }
                encoder.encode(mask_temp_message,scale,mask_temp_plain);
                mask_temp_message.clear();
                masks.push_back(mask_temp_plain);
            }
            
            
            Ciphertext encrypted_result;
            for(int i=0; i<num_queries[q];i++)
            {
                
                for(int j=0; j<d_1; j++)
                {
                    evaluator.multiply_plain(enc_queries[i], p_matrix_diagonal_plain[j], encrypted_result);
                    evaluator.relinearize_inplace(encrypted_result, relin_keys);
                    evaluator.rescale_to_next_inplace(encrypted_result);
                    
                    
                    for(int k = 0; k < logd_1; k++)
                    {
                        Ciphertext temp;
                        evaluator.rotate_vector(encrypted_result, pow(2, k), gal_keys, temp);
                        evaluator.add(encrypted_result, temp, encrypted_result);
                    }
                    
                    cout << "before mod switch" << endl;
                    evaluator.mod_switch_to_inplace(masks[j], encrypted_result.parms_id());
                    masks[j].scale() = pow(2,40);
                    cout << "after mod switch" << endl;
                    
                    evaluator.multiply_plain(encrypted_result, masks[j], encrypted_result);
                    evaluator.relinearize_inplace(encrypted_result, relin_keys);
                    evaluator.rescale_to_next_inplace(encrypted_result);
                    
                    temp_enc_fusions.push_back(encrypted_result);
                }
                Ciphertext result;
                for(int j=0; j< temp_enc_fusions.size(); j++)
                {
                    
                    if(j==0)
                        result = temp_enc_fusions[j];
                    else
                    {
                        evaluator.add_inplace(result, temp_enc_fusions[j]);
                    }
                }
                enc_fusions.push_back(result);
                
            }

            stop = high_resolution_clock::now();
            duration = duration_cast<microseconds>(stop - start);
            cout << "time to fuse: " << duration.count() / 1000.0 << " milliseconds" << endl;
            time.push_back(duration.count() / 1000.0);
            */
            //Plaintext plain_result1;
            //decryptor.decrypt(enc_fusions[0], plain_result1);
            //vector<double> result1;
            //encoder.decode(plain_result1, result1);
            //print_vector(result1, 3, 7);
            //cout << enc_fusions.size() << endl;
            //cout << endl;
            //enc_fusions.clear();
        
    
        //}
    
        //times.push_back(time);
    //}
            
    /*
    for(int i = 0; i < times.size(); i++)
    {
        cout << pow(2,i) << ": ";
        for(int j = 0; j < times[i].size(); j++)
        {
            cout << times[i][j] << " ";
        }
        cout << endl;
    }*/
}

void speed_tests1()
{

    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 8192;
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
    size_t slots = encoder.slot_count();
    
    vector<double> a1;
    Plaintext a1_plain;
    Ciphertext a1_encrypt;
    Ciphertext a1_encrypt_target;
    for(int i=0; i<slots; i++)
    {
        a1.push_back(-6.96535417e+09);
    }

    encoder.encode(a1, scale, a1_plain);
    encryptor.encrypt(a1_plain, a1_encrypt);
    double total = 0.0;
    vector<double> times;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        
        evaluator.multiply_plain(a1_encrypt, a1_plain, a1_encrypt_target);
        evaluator.relinearize_inplace(a1_encrypt_target, relin_keys);
        evaluator.rescale_to_next_inplace(a1_encrypt_target);
        
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        times.push_back(duration.count());
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    double mean = total/10000.0;
    double stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    
    cout << "avg time to plain mult: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    
    times.clear();
    total = 0.0;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        
        evaluator.multiply(a1_encrypt, a1_encrypt, a1_encrypt_target);
        evaluator.relinearize_inplace(a1_encrypt_target, relin_keys);
        evaluator.rescale_to_next_inplace(a1_encrypt_target);
        
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        times.push_back(duration.count());
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    
    
    mean = total/10000.0;
    stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    cout << "avg time to cipher mult: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    
    //cout << "avg time to plain mult: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    times.clear();
    total = 0.0;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        evaluator.rotate_vector(a1_encrypt, 1, gal_keys, a1_encrypt);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    mean = total/10000.0;
    stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    cout << "avg time to rotate: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    
    times.clear();
    total = 0.0;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        evaluator.add(a1_encrypt, a1_encrypt, a1_encrypt_target);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    mean = total/10000.0;
    stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    cout << "avg time to add: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    
}


void speed_tests2()
{

    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
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
    size_t slots = encoder.slot_count();
    
    vector<double> a1;
    Plaintext a1_plain;
    Ciphertext a1_encrypt;
    Ciphertext a1_encrypt_target;
    for(int i=0; i<slots; i++)
    {
        a1.push_back(-6.96535417e+09);
    }

    encoder.encode(a1, scale, a1_plain);
    encryptor.encrypt(a1_plain, a1_encrypt);
    double total = 0.0;
    vector<double> times;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        
        evaluator.multiply_plain(a1_encrypt, a1_plain, a1_encrypt_target);
        evaluator.relinearize_inplace(a1_encrypt_target, relin_keys);
        evaluator.rescale_to_next_inplace(a1_encrypt_target);
        
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        times.push_back(duration.count());
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    
    double mean = total/10000.0;
    double stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    
    cout << "avg time to plain mult: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    times.clear();
    
    total = 0.0;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        
        evaluator.multiply(a1_encrypt, a1_encrypt, a1_encrypt_target);
        evaluator.relinearize_inplace(a1_encrypt_target, relin_keys);
        evaluator.rescale_to_next_inplace(a1_encrypt_target);
        
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        times.push_back(duration.count());
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    
    mean = total/10000.0;
    stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    
    cout << "avg time to cipher mult: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    
    times.clear();
    total = 0.0;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        evaluator.rotate_vector(a1_encrypt, 1, gal_keys, a1_encrypt);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        times.push_back(duration.count());
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    
    mean = total/10000.0;
    stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    cout << "avg time to rotate: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    
    times.clear();
    total = 0.0;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        evaluator.add(a1_encrypt, a1_encrypt, a1_encrypt_target);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    mean = total/10000.0;
    stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    cout << "avg time to add: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    
}

void speed_tests3()
{

    cout << "setting up context" << endl;
    //all time measurement code from https://www.geeksforgeeks.org/measure-execution-time-function-cpp/
    auto start = high_resolution_clock::now();
    
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 60 }));
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
    size_t slots = encoder.slot_count();
    
    vector<double> a1;
    Plaintext a1_plain;
    Ciphertext a1_encrypt;
    Ciphertext a1_encrypt_target;
    for(int i=0; i<slots; i++)
    {
        a1.push_back(-6.96535417e+09);
    }

    encoder.encode(a1, scale, a1_plain);
    encryptor.encrypt(a1_plain, a1_encrypt);
    double total = 0.0;
    vector<double> times;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        
        evaluator.multiply_plain(a1_encrypt, a1_plain, a1_encrypt_target);
        evaluator.relinearize_inplace(a1_encrypt_target, relin_keys);
        evaluator.rescale_to_next_inplace(a1_encrypt_target);
        
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        times.push_back(duration.count());
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    
    double mean = total/10000.0;
    double stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    
    cout << "avg time to plain mult: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    times.clear();
    
    total = 0.0;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        
        evaluator.multiply(a1_encrypt, a1_encrypt, a1_encrypt_target);
        evaluator.relinearize_inplace(a1_encrypt_target, relin_keys);
        evaluator.rescale_to_next_inplace(a1_encrypt_target);
        
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        times.push_back(duration.count());
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    
    mean = total/10000.0;
    stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    
    cout << "avg time to cipher mult: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    
    times.clear();
    total = 0.0;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        evaluator.rotate_vector(a1_encrypt, 1, gal_keys, a1_encrypt);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        times.push_back(duration.count());
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    
    mean = total/10000.0;
    stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    cout << "avg time to rotate: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
    
    times.clear();
    total = 0.0;
    for(int i =0; i < 10000; i++)
    {
        start = high_resolution_clock::now();
        evaluator.add(a1_encrypt, a1_encrypt, a1_encrypt_target);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        total += duration.count();
        //cout << "time to mult: " << duration.count() / 1000.0 << " milliseconds" << endl;
    }
    mean = total/10000.0;
    stdev = 0.0;
    for(int i =0; i< 10000; i++)
    {
        stdev += pow(times[i]-mean,2);
    }
    stdev /= 10000;
    stdev = pow(stdev,0.5);
    cout << "avg time to add: " << (total/10000.0) / 1000.0 << " +- " << stdev/1000.0 << " milliseconds" << endl;
}

void encrypted_feature_fusion()
{
    //Matrix_Vector_Multiplication_Hybrid_Test();
    //Matrix_Vector_Multiplication_SIMD_Test();
    //Matrix_Vector_Multiplication_Naive_Test();
    //diagonal_1approximate_best_P_value_transpose_lambda=0.1_margin=0.75_gamma=64_reg=0.txt
    
    //degree = 3
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_degree=3strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","encrypted_results_test_lambda=0.01_margin=0.1_gamma=64_POLY3strict.txt","normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=64_POLY3strict.txt");//hybrid, poly
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_exact_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt","EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY3.txt","normalized_EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY3.txt");//hybrid, poly

//
    //degree = 2
    ////encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_2approximate_best_P_value_transpose_lambda=0.1_margin=0.75_gamma=64_reg=0.txt","encrypted_results_test_lambda=0.1_margin=0.75_gamma=64_POLY2.txt","normalized_encrypted_results_test_lambda=0.1_margin=0.75_gamma=64_POLY2.txt");//hybrid, poly
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_exact_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt","EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY2.txt","normalized_EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY2.txt",2);//hybrid, poly
    ////encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_degree=2replicated_approximate_best_P_value_transpose_lambda=0.1_margin=0.75_gamma=64_reg=0.txt", "encrypted_results_test_lambda=0.1_margin=0.75_gamma=64_POLY3replicated.txt","normalized_encrypted_results_test_lambda=0.1_margin=0.75_gamma=64_POLY2replicated.txt");//hybrid, poly
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_degree=2strict_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","encrypted_results_test_lambda=0.01_margin=0.1_gamma=64_POLY2strict.txt","normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=64_POLY2strict.txt",2);//hybrid, poly
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_approximate_best_P_value_transpose_lambda=0.01_margin=0.1_gamma=64_reg=0.txt","encrypted_results_test_lambda=0.01_margin=0.1_gamma=64_POLY2snap.txt","normalized_encrypted_results_test_lambda=0.01_margin=0.1_gamma=64_POLY2snap.txt",2);//hybrid, poly
    
    //degree = 1
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_degree=1strict_approximate_best_P_value_transpose_lambda=0.5_margin=0.5_gamma=64_reg=0.txt","encrypted_results_test_lambda=0.5_margin=0.5_gamma=64_POLY1strict.txt","normalized_encrypted_results_test_lambda=0.5_margin=0.5_gamma=64_POLY1strict.txt");//hybrid, poly
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_exact_best_P_value_transpose_lambda=0.01_margin=0.25_gamma=64_reg=0.txt","EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY1.txt","normalized_EXACT_encrypted_results_test_lambda=0.01_margin=0.25_gamma=64_POLY1.txt");//hybrid, poly
    
    //diagonal_2approximate_best_P_value_transpose_lambda=0.1_margin=0.75_gamma=64_reg=0.txt
    
    
    
    //exact learning, degree=6 inference
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_exact_best_P_value_transpose_lambda=0.25_margin=0.25_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_exact_poly6.txt","allnewdata_normalized_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_exact_poly6.txt",6);//hybrid, poly
    
    
    //exact learning, degree=6 0.05-1.0 inference
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_exact_best_P_value_transpose_lambda=0.25_margin=0.25_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_exact_poly6large.txt","allnewdata_normalized_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_exact_poly6large.txt",6);//hybrid, poly
    
    
    //exact learning, degree=3 inference
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_exact_best_P_value_transpose_lambda=0.25_margin=0.25_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_exact.txt","allnewdata_normalized_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_exact.txt",3);//hybrid, poly
    //degree=3strict learning, degree=3 inference
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_approximate_best_P_value_transpose_lambda=0.25_margin=0.1_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_poly3strict.txt","allnewdata_normalized_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_poly3strict.txt",3);//hybrid, poly
    //degree=3strict learning, degree=3 inference over 0.05-1.0 for both
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_approximate_best_P_value_transpose_lambda=0.25_margin=1.0_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.25_margin=1.0_gamma=64_poly3strictlarge.txt","allnewdata_normalized_encrypted_results_test_lambda=0.25_margin=1.0_gamma=64_poly3strictlarge.txt",3);//hybrid, poly
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_approximate_best_P_value_transpose_lambda=0.25_margin=0.75_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.25_margin=0.75_gamma=64_poly3strictlarge.txt","allnewdata_normalized_encrypted_results_test_lambda=0.25_margin=0.75_gamma=64_poly3strictlarge.txt",3);//hybrid, poly
    
    
    
    //best results yet
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_approximate_best_P_value_transpose_lambda=0.1_margin=0.5_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.1_margin=0.5_gamma=64_poly3strictlarge.txt","allnewdata_normalized_encrypted_results_test_lambda=0.1_margin=0.5_gamma=64_poly3strictlarge.txt",3);//hybrid, poly
    //best results yet, lower precision
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_approximate_best_P_value_transpose_lambda=0.1_margin=0.5_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.1_margin=0.5_gamma=64_poly3strictlarge_lowprec.txt","allnewdata_normalized_encrypted_results_test_lambda=0.1_margin=0.5_gamma=64_poly3strictlarge_lowprec.txt",3);//hybrid, poly
    //Trained on small data, evaluated on big data
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_approximate_best_P_value_transpose_lambda=0.1_margin=0.5_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.1_margin=0.5_gamma=64_poly3strictlarge_BIGDATA.txt","allnewdata_normalized_encrypted_results_test_lambda=0.1_margin=0.5_gamma=64_poly3strictlarge_BIGDATA.txt",3);//hybrid, poly
    
    //100 epochs, performs poorly - like 0.94
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/100_epochs_diagonal_approximate_best_P_value_transpose_lambda=0.25_margin=0.75_gamma=64_reg=0.txt","100_allnewdata_encrypted_results_test_lambda=0.25_margin=0.75_gamma=64_poly3strictlarge.txt","100_allnewdata_normalized_encrypted_results_test_lambda=0.25_margin=0.75_gamma=64_poly3strictlarge.txt",3);//hybrid, poly
    //100_epochs_diagonal_approximate_best_P_value_transpose_lambda=0.25_margin=0.75_gamma=64_reg=0.txt
    
    //not good enough EXACT
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_exact_best_P_value_transpose_lambda=0.1_margin=0.5_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.1_margin=0.5_gamma=64_exact.txt","allnewdata_normalized_encrypted_results_test_lambda=0.1_margin=0.5_gamma=64_exact.txt",3);//hybrid, poly
    //
    
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_approximate_best_P_value_transpose_lambda=0.1_margin=0.1_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.1_margin=0.1_gamma=64_poly3strictlarge.txt","allnewdata_normalized_encrypted_results_test_lambda=0.1_margin=0.1_gamma=64_poly3strictlarge.txt",3);//hybrid, poly
    
    //exact learning, degree=3 0.05-1.0 inference
    //encrypted_feature_fusion_polynomial_approximation_arbitrary("../data/diagonal_exact_best_P_value_transpose_lambda=0.25_margin=0.25_gamma=64_reg=0.txt","allnewdata_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_exact_poly3large.txt","allnewdata_normalized_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_exact_poly3large.txt",3);//hybrid, poly
    
    
    //encrypted_feature_fusion_goldschmidt();//hybrid, gold
    
    encrypted_feature_fusion_polynomial_approximation_arbitrary("../big_data/diagonal_approximate_best_P_value_transpose_lambda=0.25_margin=0.25_gamma=64_reg=0.txt","000_allnewdata_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_poly3strictlarge.txt","000_allnewdata_normalized_encrypted_results_test_lambda=0.25_margin=0.25_gamma=64_poly3strictlarge.txt",3);//hybrid, poly
    
    //encrypted_feature_fusion_SIMD_polynomial();
    //encrypted_feature_fusion_SIMD_goldschmidt();//simd, gold
    //speed_tests1();
    //speed_tests2();
    //speed_tests3();
}
