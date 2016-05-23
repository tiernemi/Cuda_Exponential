CXX= g++
NVCC= /usr/local/cuda/bin/nvcc
LINK= nvcc
DEL_FILE= rm -f

#Flags
#PARALLEL	= -fopenmp
#DEFINES		= -DWITH_OPENMP
CFLAGS		= -W -Wall -lcuda $(PARALLEL) $(DEFINES)
CXXFLAGS    = -lGL -lglut -lpthread -llibtiff  -Og -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops -W -Wall -lcuda $(PARALLEL) $(DEFINES) -std=c++11

NVCCFLAGS	= -O5 -DWITH_MY_DEBUG -std=c++11 -arch=sm_35 --relocatable-device-code true -lcudadevrt --use_fast_math 
LIBS		= $(PARALLEL)
INCPATH		= /usr/include/

####### Files
SRC=integral.cu main.cpp
OBJ=integral.o main.o
SOURCES=$(SRC)
OBJECTS=$(OBJ)
TARGET= integral

all: $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $(TARGET) -I$(INCPATH) -lcudadevrt

main.o: main.cpp integral.hpp
	$(CXX) -c $< $(CXXFLAGS) -I$(INCPATH)

# seperate compilation
integral.o: integral.cu integral.hpp
	$(NVCC) -c $< $(NVCCFLAGS) -I$(INCPATH)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)

remote:
	mkdir -p temp
	cp *.hpp temp
	cp *.cpp temp
	cp *.cu temp
	cp Makefile temp
	tar -cvf temp.tar.gz temp
	scp temp.tar.gz CUDA01\:~
	ssh -A CUDA01 "source /etc/profile && ~/.bash_profile && tar -xvf temp.tar.gz && cd temp && $$(make)"
	
