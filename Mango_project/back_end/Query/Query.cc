#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <deque>
#include <cmath>
#include "Clothes.hh"
using namespace std;

// This should be calculated with RM
const int ConstColor = 1, ConstType = 1, ConstOutfit = 3;

void init(vector<vector<int>> & DPcolor, vector<vector<int>> & DPtype, map<pair<int, int>, int> & DPoutfit,
        map<string, int> & ColorToNumeric, map<string, int> & TypeToNumeric,
        vector<Clothes> & ClothesDataBase, map<int, Clothes> & SearchByNumericId,
        map<string, int> & IdToNumericId, fstream & in) {
    // first read the clothes
    in.open("products.txt", ios::in);  // read from the file where the products are
    string id, color, sex, age, cathegory, aggregatedFamily;
    int numberProducts;                // Number of product from the input file
    in >> numberProducts;
    int colorCount, typeCount;
    colorCount = typeCount = 1;
    for (int i = 0; i < numberProducts; ++i) {
        getline(in, id);               // We read an extra time to flush getline function
        getline(in, id);
        getline(in, color);
        getline(in, sex);
        getline(in, age);
        getline(in, cathegory);
        getline(in, aggregatedFamily);
        ClothesDataBase.push_back(Clothes(id, color, sex, age, cathegory, aggregatedFamily));

        // Set bijections
        IdToNumericId[id] = i;  
        SearchByNumericId[i] = ClothesDataBase[ClothesDataBase.size()-1];
        if (not ColorToNumeric.count(color)) ColorToNumeric[color] = colorCount++;
        if (not TypeToNumeric.count(aggregatedFamily)) TypeToNumeric[aggregatedFamily] = typeCount++;
    }
    in.close();

    // Now read the outfits building the DP
    in.open("outfits.txt", ios::in);
    int numberOutfits;      // Number of good outfits
    in >> numberOutfits;
    string s;
    getline(in, s);         // flush input
    while (getline(in, s)) {
        
        // First get the input as strings
        deque<string> input;
        string curr = "";
        for (int i = 0; i < static_cast<int>(s.size()); ++i) {
            if (s[i] == ' ') {
                input.push_back(curr);
                curr = "";
            }
            else {
                curr += s[i];
            }
        }
        input.pop_front(); // We do not really care about the ID, just the elements
        
        // Transform strings to ints (bijection)
        vector<int> ids;
        for (int i = 0; i < static_cast<int>(input.size()); ++i) {
            ids.push_back(IdToNumericId[input[i]]);
        }
        

        // set DP of colors
        for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(ids.size()); ++j) {
                DPcolor[ColorToNumeric[ SearchByNumericId[ids[i]].getColor() ]][ColorToNumeric[ SearchByNumericId[ids[j]].getColor() ]]++;
                DPcolor[ColorToNumeric[ SearchByNumericId[ids[j]].getColor() ]][ColorToNumeric[ SearchByNumericId[ids[i]].getColor() ]]++;
            }
        }
        
        // set DP of types
        for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(ids.size()); ++j) {
                DPtype[TypeToNumeric[ SearchByNumericId[ids[i]].getAggregatedFamily() ]][TypeToNumeric[ SearchByNumericId[ids[j]].getAggregatedFamily() ]]++;
                DPtype[TypeToNumeric[ SearchByNumericId[ids[j]].getAggregatedFamily() ]][TypeToNumeric[ SearchByNumericId[ids[i]].getAggregatedFamily() ]]++;
            }   
        }

        // set DP of outfits
        for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
            for (int j = i + 1; j < static_cast<int>(ids.size()); ++j) {
                DPoutfit[make_pair(ids[i],ids[j])]++;
            }   
        }
    }

    in.close();

}


int main() {
    fstream in, out;   // Aux variable to set input and output files
    out.open("IdsOutput.txt", ios::out);

    
    vector<vector<int>> DPcolor(200, vector<int> (200));
    vector<vector<int>> DPtype(200, vector<int> (200));
    map<pair<int, int>, int> DPoutfit;

    map<string, int> IdToNumericId;                                  // set a numeric Id for each Clothe (Speed up algorithm)
    map<int, Clothes> SearchByNumericId;                             // Store Clothes by ID
    map<string, int> ColorToNumeric;                                 // More bijections to speed up the algorithm
    map<string, int> TypeToNumeric;
    vector<Clothes> ClothesDataBase;                                 // Stores the available clothes
    
    init(DPcolor, DPtype, DPoutfit, ColorToNumeric, TypeToNumeric, ClothesDataBase, SearchByNumericId, IdToNumericId, in); 
    // Starts the program, it read both input files and built the Data Structures

    // Now read the clothes that the user want to use
    in.open("IdsToQuery.txt", ios::in);
    vector<int> Ids;
    string s;
    while (in >> s) {
        Ids.push_back(IdToNumericId[s]);
    }

    // Set how many of basic outfit (bottom, top, outerware) is missing
    vector<bool> kind(3);
    for (int i = 0; i < static_cast<int>(Ids.size()); ++i) {
        for (int j = 0; j < 3; ++j) {
            kind[j] = kind[j] or SearchByNumericId[Ids[i]].getType()[j];
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (kind[i]) continue;
        // In this point we need to add clothes of type i
        int arg = 0, best = -1e9; // which clothe and its value

        // We iterate though all element and get the best one based on entropy
        for (int j = 0; j < static_cast<int>(ClothesDataBase.size()); ++j) {
            int valueColor = 0;
            int valueType = 0;
            int valueOutfit = 0;
            for (int k = 0; k < static_cast<int>(Ids.size()); ++k) {
                valueColor += log(1 + DPcolor[ColorToNumeric[ ClothesDataBase[j].getColor() ]][ColorToNumeric[ ClothesDataBase[Ids[k]].getColor()]]);
                valueType += log(1 + DPtype[TypeToNumeric[ ClothesDataBase[j].getAggregatedFamily() ]][TypeToNumeric[ ClothesDataBase[Ids[k]].getAggregatedFamily() ]]);
                valueOutfit += log(1 + DPoutfit[make_pair(j, Ids[k])]);
            }

            if (ConstColor * valueColor + ConstType * valueType + ConstOutfit * valueOutfit > best) {
                arg = i;
                best = ConstColor * valueColor + ConstType * valueType + ConstOutfit * valueOutfit;
            }
        }

        // Update values
        kind[i] = true;
        Ids.push_back(arg);
    }
    
    // To Do: Add accessories


    // Print the selected values
    for (int i = 0; i < static_cast<int>(Ids.size()); ++i) {
        for (int j = 0; j < static_cast<int>(ClothesDataBase.size()); ++j) {
            if (IdToNumericId[ClothesDataBase[j].getId()] == Ids[i]) {
                out << ClothesDataBase[j].getId() << endl;
            }
        }
    }


    in.close();
    out.close();
}