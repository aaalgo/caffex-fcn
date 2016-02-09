CC=g++
CFLAGS += -O3 -g 
CXXFLAGS += -std=c++11 -O3 -fopenmp -g -I/opt/caffe-fcn/include
LDFLAGS += -static -fopenmp -L/opt/caffe-fcn/lib -static
#LDLIBS +=  -lxgboost /usr/local/lib/dmlc_simple.o -lrabit -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive -lproto -lprotobuf -lsnappy -lgflags -lglog -lleveldb -llmdb -lunwind -lhdf5_hl -lhdf5 -lopencv_features2d -lopencv_imgproc -lopencv_imgcodecs -lopencv_flann -lopencv_core -lopencv_hal -lIlmImf -lippicv -lboost_timer -lboost_chrono -lboost_program_options -lboost_log -lboost_log_setup -lboost_thread -lboost_filesystem -lboost_system -lopenblas -ljpeg -ltiff -lpng -ljasper -lwebp -lpthread -lz -lm -lrt -ldl
LDLIBS =  -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive \
	 -lopencv_ml -lopencv_imgproc -lopencv_highgui -lopencv_core \
	 -lboost_timer -lboost_chrono -lboost_thread -lboost_filesystem -lboost_system -lboost_program_options \
	 -lprotoc -lprotobuf -lglog -lgflags -lleveldb -llmdb \
	 -lhdf5_hl -lhdf5 \
	 -ljpeg -lpng -ltiff -lgif -ljasper  \
	 -lsnappy -lz \
	 -lopenblas \
	 -lunwind -lrt -lm -ldl

PROGS = run caffex-extract	caffex-predict batch-resize import-images

all:	$(PROGS)

caffex-extract:	caffex-extract.cpp caffex.cpp

caffex-predict:	caffex-predict.cpp caffex.cpp

caffex-compare:	caffex-compare.cpp caffex.cpp

run:	run.cpp caffex.o

batch-resize:	batch-resize.cpp
