CUDA_INSTALL_PATH ?= /usr/local/cuda
CC=g++
NV=$(CUDA_INSTALL_PATH)/bin/nvcc
LD=$(CUDA_INSTALL_PATH)/bin/nvcc
CCFLAGS=-Isrc -I$(CUDA_INSTALL_PATH)/include -std=c++11 -Wall
# NVARCH=--gpu-architecture=compute_61 --gpu-code=sm_61
NVFLAGS=-Isrc -std=c++11 $(NVARCH) --compiler-options -Wall --resource-usage
NVFLAGSDEP=-Isrc -std=c++11 $(NVARCH)
LDFLAGS=$(NVARCH) -lcuda -lpng
debug : CCFLAGS += -g
release : CCFLAGS += -O3 -DNDEBUG
release : LDFLAGS += -O3
SRC_DIR:=src
INT_DIR_DEBUG:=build/debug
INT_DIR_RELEASE:=build/release
OUT_DIR=bin
TARGET_DEBUG=programd
TARGET_RELEASE=program

SOURCES := $(shell find $(SRC_DIR) -name *.cpp)
CUSOURCES := $(shell find $(SRC_DIR) -name *.cu)
SRCDIRS := $(shell find $(SRC_DIR) -type d)

DIRS_DEBUG = $(SRCDIRS:$(SRC_DIR)%=$(INT_DIR_DEBUG)%)
DIRS_RELEASE = $(SRCDIRS:$(SRC_DIR)%=$(INT_DIR_RELEASE)%)

OBJS_DEBUG = $(SOURCES:$(SRC_DIR)/%.cpp=$(INT_DIR_DEBUG)/%.o)
OBJS_RELEASE = $(SOURCES:$(SRC_DIR)/%.cpp=$(INT_DIR_RELEASE)/%.o)
CUOBJS_DEBUG = $(CUSOURCES:$(SRC_DIR)/%.cu=$(INT_DIR_DEBUG)/%.cu.o)
CUOBJS_RELEASE = $(CUSOURCES:$(SRC_DIR)/%.cu=$(INT_DIR_RELEASE)/%.cu.o)
DEPS_DEBUG = $(SOURCES:$(SRC_DIR)/%.cpp=$(INT_DIR_DEBUG)/%.d)
DEPS_RELEASE = $(SOURCES:$(SRC_DIR)/%.cpp=$(INT_DIR_RELEASE)/%.d)
CUDEPS_DEBUG = $(CUSOURCES:$(SRC_DIR)/%.cu=$(INT_DIR_DEBUG)/%.cud)
CUDEPS_RELEASE = $(CUSOURCES:$(SRC_DIR)/%.cu=$(INT_DIR_RELEASE)/%.cud)

all: debug

debug: $(OUT_DIR)/$(TARGET_DEBUG)

release: $(OUT_DIR)/$(TARGET_RELEASE)

$(OUT_DIR)/$(TARGET_DEBUG) : $(OBJS_DEBUG) $(CUOBJS_DEBUG) | $(DIRS_DEBUG) $(OUT_DIR)
	$(LD) $^ $(LDFLAGS) -o $@

$(OUT_DIR)/$(TARGET_RELEASE) : $(OBJS_RELEASE) $(CUOBJS_RELEASE) | $(DIRS_RELEASE) $(OUT_DIR)
	$(LD) $^ $(LDFLAGS) -o $@

$(INT_DIR_DEBUG)/%.o : $(SRC_DIR)/%.cpp $(INT_DIR_DEBUG)/%.d | $(DIRS_DEBUG)
	$(CC) $< $(CCFLAGS) -c -o $@

$(INT_DIR_RELEASE)/%.o : $(SRC_DIR)/%.cpp $(INT_DIR_RELEASE)/%.d | $(DIRS_RELEASE)
	$(CC) $< $(CCFLAGS) -c -o $@

$(INT_DIR_DEBUG)/%.cu.o : $(SRC_DIR)/%.cu $(INT_DIR_DEBUG)/%.cud | $(DIRS_DEBUG)
	$(NV) $< $(NVFLAGS) -c -o $@

$(INT_DIR_RELEASE)/%.cu.o : $(SRC_DIR)/%.cu $(INT_DIR_RELEASE)/%.cud | $(DIRS_RELEASE)
	$(NV) $< $(NVFLAGS) -c -o $@

$(INT_DIR_DEBUG)/%.d : $(SRC_DIR)/%.cpp | $(DIRS_DEBUG)
	@$(CC) $(CCFLAGS) $< -MM -MP |\
		sed 's=\($(*F)\)\.o[ :]*=$(@D)/\1.o $@ : =g;'\
		> $@

$(INT_DIR_RELEASE)/%.d : $(SRC_DIR)/%.cpp | $(DIRS_RELEASE)
	@$(CC) $(CCFLAGS) $< -MM -MP |\
		sed 's=\($(*F)\)\.o[ :]*=$(@D)/\1.o $@ : =g;'\
		> $@

$(INT_DIR_DEBUG)/%.cud : $(SRC_DIR)/%.cu | $(DIRS_DEBUG)
	@$(NV) $(NVFLAGSDEP) $< -M |\
		sed 's=\($(*F)\)\.o[ :]*=$(@D)/\1.o $@ : =g;'\
		> $@

$(INT_DIR_RELEASE)/%.cud : $(SRC_DIR)/%.cu | $(DIRS_RELEASE)
	@$(NV) $(NVFLAGSDEP) $< -M |\
		sed 's=\($(*F)\)\.o[ :]*=$(@D)/\1.o $@ : =g;'\
		> $@

$(DIRS_DEBUG) $(DIRS_RELEASE) $(OUT_DIR):
	@mkdir -p $@

clean:
	@rm -rf $(INT_DIR_DEBUG)
	@rm -rf $(INT_DIR_RELEASE)
	@rm -rf $(OUT_DIR)

.PHONY: all debug release clean
.SECONDARY: $(OBJS_DEBUG) $(OBJS_RELEASE) $(CUOBJS_DEBUG) $(CUOBJS_RELEASE) $(DEPS_DEBUG) $(DEPS_RELEASE) $(CUDEPS_DEBUG) $(CUDEPS_RELEASE)

simpleTest: all
	./bin/programd < simpleTest.in > agents_positions.out

-include $(DEPS) $(CUDEPS)

