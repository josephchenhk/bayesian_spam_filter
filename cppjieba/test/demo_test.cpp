/*************************************************  
     
    Author: Joseph Chen
     
    Date: 2017-11-02
     
    Description: C++ version of Bayesian spam filter.
     
**************************************************/ 

#include "cppjieba/Jieba.hpp"
#include <algorithm>
#include <fstream>
#include <time.h>
#include <stdio.h>
#include <string>
#include <ctime> // time_t

using namespace std;

const char* const DICT_PATH = "../dict/jieba.dict.utf8";
const char* const HMM_PATH = "../dict/hmm_model.utf8";
//const char* const USER_DICT_PATH = "../dict/user.dict.utf8";
const char* const USER_DICT_PATH = "../jieba/userdict.txt";
const char* const IDF_PATH = "../dict/idf.utf8";
const char* const STOP_WORD_PATH = "../dict/stop_words.utf8";

vector<string> filter(string, vector<string>);
vector<string> grab_words(string);
double p_cal(string);
map<string,double> analyse_onemsg(string);
map<string,int> load_wordDic(string);
vector<string> split(string, string);
double total_probability(map<string,double>, int);
double multiply_values(vector<double>);
map<string,int> learn(string);
void mk_word_dict(string, map<string, int>);
bool replace(std::string&, const std::string&, const std::string&);
vector<vector<double> > analyse_testmsg (string);
void join_raw_data(string, string, vector<string>);
double predict_spam_prob(string);
bool predict_spam(string, double);
void test_one_line_msg(string);
void test_learning();
void test_file_msg(vector<double>);

// Jieba lib
cppjieba::Jieba jieba(DICT_PATH,
        HMM_PATH,
        USER_DICT_PATH,
        IDF_PATH,
        STOP_WORD_PATH);

// main path        
string homepath = "..";

//path for learning
string pathl = homepath + "/data/learn";
string pathln= pathl + "/normal.txt";         // material to be learnt
string pathls= pathl + "/spam.txt";           // material to be learnt

//path for testing
string patht = homepath + "/data/test";
string pathtn= patht + "/testnormal.txt";     // material to be tested
string pathts= patht + "/testspam.txt";       // material to be tested

//path for saving
string paths = homepath + "/data/save";
string pathsn= paths + "/wnormal.txt";        // keywords and freq //extracted from normal
string pathss= paths + "/wspam.txt";          // keywords and freq //extracted from spam

//path for raw data
string pathraw = homepath + "/data/raw/";     // raw data directory

// Dictionaries define as global
map<string,int> nwordDic;
map<string,int> swordDic;

int main(int argc, char** argv) {
    cout << "<< Bayesian spam filter >>" << endl; 
    
    // Load dictionaries
    nwordDic = load_wordDic(pathsn.c_str());
    swordDic = load_wordDic(pathss.c_str());
        
    // test predict_spam 
    string msg;
    msg = "S B";   
    cout<< "\nMessage [" << msg << "] probability of spam: " << predict_spam_prob(msg) << endl;
    msg = "想赢。搜公纵號〔妞姐看牌〕";       
    cout<< "\nMessage [" << msg << "] probability of spam: " << predict_spam_prob(msg) << endl;
    
    return EXIT_SUCCESS;
}

/*    
int main(int argc, char** argv) {
    cout << "Bayesian spam filter" << endl;
    
    // Prepare learning and testing data
    vector<string> normal_learn;
    vector<string> normal_test;
    normal_learn.push_back("raw_normal_20170901_20170905.tsv");
    normal_learn.push_back("raw_normal_20170906_20170910.tsv");
    normal_learn.push_back("raw_normal_20170911_20170920.tsv");
    normal_learn.push_back("raw_normal_20170921_20170930.tsv");
    normal_learn.push_back("raw_normal_20171001_20171010.tsv");
    normal_test.push_back("raw_normal_20171011_20171020.tsv");
    normal_test.push_back("raw_normal_20171021_20171031.tsv");
    join_raw_data("normal", "learn", normal_learn);
    join_raw_data("normal", "test", normal_test);
    
    vector<string> spam_learn;
    vector<string> spam_test;
    spam_learn.push_back("raw_spam_20170901_20170905.tsv");
    spam_learn.push_back("raw_spam_20170906_20170910.tsv");
    spam_learn.push_back("raw_spam_20170911_20170920.tsv");
    spam_learn.push_back("raw_spam_20170921_20170930.tsv");
    spam_learn.push_back("raw_spam_20171001_20171010.tsv");
    spam_test.push_back("raw_spam_20171011_20171020.tsv");
    spam_test.push_back("raw_spam_20171021_20171031.tsv");
    join_raw_data("spam", "learn", spam_learn);
    join_raw_data("spam", "test", spam_test);  
    
    // Load dictionaries
    nwordDic = load_wordDic(pathsn.c_str());
    swordDic = load_wordDic(pathss.c_str());
       
    // learning
    test_learning();
    
    // one line message test   
    string msg;
    cout << "Example of Normal Message" << endl;
    msg = "我是好人";    
    test_one_line_msg(msg);
    cout << "Example of Normal Message" << endl;
    msg = "对子收收完美";   
    test_one_line_msg(msg);
    cout << "Example of Spam Message" << endl;
    msg = "S B";   
    test_one_line_msg(msg);
    cout << "Example of Spam Message" << endl;
    msg = "想赢。搜公纵號〔妞姐看牌〕";   
    test_one_line_msg(msg); 
    
    // test predict_spam and predict_spam_prob
    cout<< "Message [" << msg << "] is_spam probability: " << predict_spam_prob(msg) << endl;
    cout<< "Under threshold ["<< 0.09 << "] this messeage is classified as spam: " << predict_spam(msg,0.09)<<endl;
       
    // message in file test
    std::vector<double> threshold;
    threshold.push_back(0.01);
    threshold.push_back(0.03);
    threshold.push_back(0.05);
    threshold.push_back(0.07);
    threshold.push_back(0.09);
    test_file_msg(threshold);
     
    return EXIT_SUCCESS;
}
*/

double predict_spam_prob(string msg){
    //param msg: input message to be tested
    //return prob: probability of message being classified as spam
    double prob;
    map<string, double> p_dic;
    p_dic = analyse_onemsg(msg);  
    prob = total_probability(p_dic, p_dic.size());
    return prob;
}

bool predict_spam(string msg, double threshold){
    //param msg: input message to be tested
    //return prob: probability of message being classified as spam
    double prob;
    map<string, double> p_dic;
    p_dic = analyse_onemsg(msg);  
    prob = total_probability(p_dic, p_dic.size());
    if(prob>=threshold){
        return true;
    }else{
        return false;
    }
}

void test_learning(){
    clock_t start = clock();  // tic
    cout << "[1]-------------------" << endl;
    cout << "Start learning ..." << endl;
    nwordDic = learn(pathln);
    mk_word_dict(pathsn, nwordDic);
    swordDic = learn(pathls);
    mk_word_dict(pathss, swordDic);    
    clock_t stop = clock();  // toc
    double elapsed = (double)(stop - start) * 1.0 / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds.\n", elapsed);
    cout << "----------------------\n" << endl;
} 

void test_one_line_msg(string msg){    
    //double prob;
    //map<string, double> p_dic;
    //p_dic = analyse_onemsg(msg);  
    //prob = total_probability(p_dic, p_dic.size());
    
    double prob = predict_spam_prob(msg);
    cout << "[2]-------------------" << endl;
    cout << "Msg: " << msg << endl;
    cout << "prob of spam: "<< prob << endl;
    cout << "----------------------\n" << endl;
}

void test_file_msg(vector<double> threshold){
    vector<vector<double> > msgp1_vec;
    vector<vector<double> > msgp2_vec;
    int counter1;
    int counter2;
    double th;
    double value;
    double normal_fail;
    double spam_correct;
    
    vector<double>::iterator it;
    vector<double>::iterator it1;
    vector<double>::iterator it2;
    
    msgp1_vec = analyse_testmsg(pathtn); 
    msgp2_vec = analyse_testmsg(pathts);   
    
    cout << "[3]-------------------" << endl;
    cout<< "number of normal msg analysed: "<< msgp1_vec[1][0] <<endl;
    cout<< "number of spam msg analysed: "<< msgp2_vec[1][0] <<endl;
    for(it=threshold.begin(); it!=threshold.end(); it++){
        th = *it;
        counter1 = 0;
        counter2 = 0;
        for(it1=msgp1_vec[0].begin(); it1!=msgp1_vec[0].end(); it1++){
            value = *it1;
            if(value>=th){
                counter1 = counter1 + 1;
            }
        }
        
        for(it2=msgp2_vec[0].begin(); it2!=msgp2_vec[0].end(); it2++){
            value = *it2;
            if(value>=th){
                counter2 = counter2 + 1;
            }
        }
        //cout<< counter1 << "; " << msgp1_vec[1][0] << " | "<<counter2 << ";" << msgp2_vec[1][0] <<endl;
        normal_fail = (double)counter1/msgp1_vec[1][0];
        spam_correct = (double)counter2/msgp2_vec[1][0];
        cout<<"Threshold: "<< th << endl;
        cout<<"Normal regarded as spam = " << counter1 << "; " << "normal msg fail percentage = " << normal_fail<<endl;
        cout<<"Spam detected as spam = " << counter2 << "; " << "spam msg correct percentage = " << spam_correct<<endl;
        cout << "\n" <<endl;
    }
    cout << "----------------------\n" << endl;
}

vector<string> filter(string to_remove, vector<string> seg_set){
    vector<string> return_seg_vec;
    vector<string>::iterator it;       // declare an iterator to a vector of strings
    for(it=seg_set.begin(); it!=seg_set.end(); it++){
        string value = *it;
        if(value!=to_remove){
            return_seg_vec.push_back(value);
        }
    }
    return return_seg_vec;
}

vector<string> grab_words(string msg){
    vector<string> seg_list_org;
    msg.erase(std::remove(msg.begin(), msg.end(), '\n'), msg.end()); // replace "\n" with ""
    msg.erase(std::remove(msg.begin(), msg.end(), '\r'), msg.end()); // replace "\r" with ""
    std::transform(msg.begin(), msg.end(), msg.begin(), ::tolower);  // ignore upper and lower case    
    jieba.Cut(msg, seg_list_org, true); // separate into a list of keywords using jieba  
    set<string> seg_set(seg_list_org.begin(), seg_list_org.end());
    vector<string> seg_set_vec(seg_set.begin(), seg_set.end()); // for repeated keywords, only take one
    seg_set_vec = filter(" ", seg_set_vec);  // remove space   
    return seg_set_vec;
}

double p_cal(string word){
    // p(s|w) = 0.4 if it does not exist in spam and normal database
    // p(s|w) < 0.99 avoid extreme value (if it only exist in spam database) 
    // p(s|w) > 0.01 avoid extreme value (if it only exist in normal database)        
    double minp = 0.01;
    double maxp = 0.99;
    int nws;
    int nwn;
    double psw;
    bool word_in_swordDic, word_in_nwordDic;   
    word_in_swordDic = (bool)swordDic.count(word);
    word_in_nwordDic = (bool)nwordDic.count(word);  
    
    if (word_in_swordDic && word_in_nwordDic){
        nws = swordDic.at(word);         // frequency in spam msg
        nwn = nwordDic.at(word);         // frequency in normal msg
        if (nws+nwn<=3){
            psw = 0.4;
        }else{
            psw = (double)nws/(nws+nwn); // probability that the msg is spam given the msg contains the word
        }
    }else if ((!word_in_swordDic) && word_in_nwordDic){
        nwn = nwordDic.at(word);
        if (nwn<=2){
            psw = 0.4;
        }else{
            psw = minp;
        }   
    }else if (word_in_swordDic && (!word_in_nwordDic)){
        nws = swordDic.at(word);
        if (nws<=2){
            psw = 0.4;
        }else{
            psw = maxp;
        }
    }else{
        psw = 0.4;
    }
      
    if (psw>maxp){
        psw = maxp;
    }else if (psw<minp){
        psw = minp;
    }  
    return psw;
} 

map<string,int> load_wordDic(string path){  
    std::ifstream file(path.c_str());
    string str; 
    string data_str;
    int data_freq;
    vector<string> data;
    map<string,int> nwordDic;
    while (std::getline(file, str) ){ 
        data = split(str, " = ");
        data_str = data[0];
        data_freq = std::atoi(data[1].c_str());
        nwordDic[data_str] = data_freq;
    } 
    return nwordDic;
}

vector<string> split(string str, string sep){
    char* cstr=const_cast<char*>(str.c_str());
    char* current;
    vector<std::string> arr;
    current=strtok(cstr,sep.c_str());
    while(current != NULL){
        arr.push_back(current);
        current=strtok(NULL, sep.c_str());
    }
    return arr;
}

double total_probability(map<string,double> p_dic, int n){
    // calculate probability using input keyword probability dictionary, up to n highest term'
    // probability = times all p_dic/ (times all p_dic + times all 1-p_dic)

    // turn dictionary into list, remains the first n highest term
    double probability;
    vector<double> p_dic_list;
    for(map<string,double>::iterator it = p_dic.begin(); it != p_dic.end(); ++it) {
        p_dic_list.push_back(it->second);
    }
    // sort the probabilities from highest to lowest (reversely)
    std::sort(p_dic_list.rbegin(), p_dic_list.rend());

    // prevent n exceeds the range
    if ((unsigned)n> p_dic_list.size()){
        n = p_dic_list.size(); 
    }
    p_dic_list.resize(n);
  
    // 1-p_dic
    vector<double> one_minus_p_dic_list;
    double p_value;
    for (std::vector<double>::iterator it = p_dic_list.begin() ; it != p_dic_list.end(); ++it){
        p_value = *it;
        one_minus_p_dic_list.push_back(1-p_value);
    }

    probability = multiply_values(p_dic_list)/(multiply_values(p_dic_list)+multiply_values(one_minus_p_dic_list));  
    return probability;
}

double multiply_values(vector<double> listtomultiply){
    double output;
    double value;
    output = 1.0;
    for (std::vector<double>::iterator it = listtomultiply.begin() ; it != listtomultiply.end(); ++it){
        value = *it;
        output = output*value;
    }
    return output;
}

map<string,int> learn(string fpath){
    // learn msg from txt 
    // grab keywords and update the keyword frequency dictionary 
    map<string,int> dicttosave;
    std::ifstream file(fpath.c_str());
    string value;
    string line;
    vector<string> seg_set;
    int counter;
    counter = 0;
    while (std::getline(file, line)){
        counter = counter + 1;
        // grab keywords, returns a set of keywords
        seg_set = grab_words(line);                
        //update the keyword dictionary     
        for (std::vector<string>::iterator it = seg_set.begin() ; it != seg_set.end(); ++it){
            value = *it;
            if (dicttosave.count(value)){
                dicttosave[value] = dicttosave[value] + 1;
            }else{
                dicttosave[value] = 1;
            }
        }
    } 
    cout<< "number of msg learnt: " <<counter <<endl;
    return dicttosave;
}

void mk_word_dict(string fpath, map<string, int> dicttoexport){
    // export grapped keywords with frequency into txt
    ofstream word;
    word.open(fpath.c_str());
    string key;
    int value;
    for(map<string, int>::iterator it = dicttoexport.begin(); it != dicttoexport.end(); ++it) {
        key = it->first;
        value = it->second;
        word<< key << " = " << value <<endl;
    }
    word.close();
}

bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

vector<vector<double> > analyse_testmsg(string testfile){
    //testfile is path of testing msg data base
    vector<double> msgp;
    vector<double> counter_vec;
    vector<vector<double> > results;
    map<string,double> p_dic;
    double counter;
    double t_probability;
    
    clock_t tic, toc;
    double cal_time;
    
    // read testing data
    std::ifstream infile(testfile.c_str());
    
    // make output file (logfile)    
    replace(testfile, "testnormal", "log_testnormal"); 
    replace(testfile, "testspam", "log_testspam"); 
    ofstream logfile;
    logfile.open(testfile.c_str());
    
    // analyse each msg  
    string line;
    counter = 0.0;
    
    tic = clock();
    
    while (std::getline(infile, line) ){       
        ++counter;
        p_dic = analyse_onemsg(line);
        t_probability = total_probability(p_dic, p_dic.size());
        msgp.push_back(t_probability);
        
        // export to logfile
        logfile<<"probability = "<<t_probability<<"\n"<<line<<"\n"<<p_dic<<endl<<endl;
        //cout<<"probability = "<<t_probability<<"\n"<<line<<"\n"<<p_dic<<endl<<endl;
               
        // Output to screen to track progress
        if((int)counter%500000==0){
            toc = clock();
            cal_time = (double)(toc - tic) * 1.0 / CLOCKS_PER_SEC;
            cout << counter << ": " << cal_time << " s"  << endl; 
            tic = clock();
        }   
    }
  
    logfile.close();
    counter_vec.push_back(counter);
    results.push_back(msgp);
    results.push_back(counter_vec);
    return results;
}

map<string,double> analyse_onemsg(string line){   
    // analyse one msg, return spam probability.
    vector<string> seg_set;
    double p_cal_value ;   
    seg_set = grab_words(line) ;       // separate into keywords    
    map<string,double> p_dict;         // keyword probability dictionary  
    vector<string>::iterator it;       // declare an iterator to a vector of strings     
    for(it=seg_set.begin(); it!=seg_set.end(); it++){
        string values = *it;
        p_cal_value = p_cal(values);
        p_dict.insert ( std::pair<string,double>(values, p_cal_value) ); 
    }  
    return p_dict;
}

void join_raw_data(string category, string purpose, vector<string> data_name){
    // param category: "normal" or "spam"
    // param purpose: "learn" or "test"
    // param data_name: suffix name of files
    string value;
    string raw_data_file;
    std::ifstream infile;
    std::ofstream outfile;
    double counter;
    if(category=="normal"){
        if(purpose=="learn"){
            outfile.open(pathln.c_str());
        }else if(purpose=="test"){
            outfile.open(pathtn.c_str());
        }
    }else if(category=="spam"){
        if(purpose=="learn"){
            outfile.open(pathls.c_str());
        }else if(purpose=="test"){
            outfile.open(pathts.c_str());
        }
    }
    for (std::vector<string>::iterator it = data_name.begin() ; it != data_name.end(); ++it){
        value = *it;
        raw_data_file = pathraw + value;
        cout << raw_data_file << endl;
        std::ifstream infile(raw_data_file.c_str());
        counter = 0.0;
        while (std::getline(infile, value) ){
            ++counter;
            /*
            if(counter<=3.0){
                cout << "(" << counter << ") msg: " << value << endl;
                continue;
            }*/
            outfile << value << endl;
        }
    }
    infile.close();
    outfile.close();
}