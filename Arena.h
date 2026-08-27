//
// Created by athos on 26/08/2026.
//

#ifndef GENALGNEURALNETWORK2_ARENA_H
#define GENALGNEURALNETWORK2_ARENA_H

using byte = unsigned char; // 1 byte

class Arena {
public:
    explicit Arena(const int size) : memorySize{size} {
        data = new byte[size];
        dataTail = 0;
    }

    void* a_malloc(const int blockSize, const int unitSize) {
        if (dataTail+blockSize > memorySize) {
            return nullptr;
        }

        void* dataPointer = (data+dataTail);
        dataTail += blockSize * unitSize;
        data+=blockSize;
        return dataPointer;
    }

    void a_free() const {
        delete [] data;
    }

private:
    byte* data;
    int dataTail;
    int memorySize;
};


#endif //GENALGNEURALNETWORK2_ARENA_H
