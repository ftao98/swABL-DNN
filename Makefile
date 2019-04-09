# ======================================
# || Author: WuZheng from USTC-ACSA   ||
# || Email : zhengwu@mail.ustc.edu.cn ||
# || Data  : 2019-02-25               ||
# ======================================

# 指定编译链接的command和options
GCC = gcc
G++ = g++
CFLAGS += -O2 -W -fPIC 
LDFLAGS += -lm

# release or debug
# CFLAGS += -g

# 依赖文件及相关路径
INC_PATH += $(INIT_PATH)/include
LIB_PATH += $(INIT_PATH)
CFLAGS += $(INC_PATH:%=-I%)
LDFLAGS += $(LIB_PATH:%=-L%)
LDFLAGS += $(LIB_PATH:%=-Wl,-rpath=%) # 通过-Wl,-rpath=使得execute记住链接库的路径

# 文件 & 相关路径
INIT_PATH	= .
SRC_PATH	= $(shell find $(INIT_PATH)/src -maxdepth 3 -type d)
TOOL_PATH	= $(shell find $(INIT_PATH)/tool -maxdepth 3 -type d)

SRC 		= $(foreach dir,$(SRC_PATH),$(wildcard $(dir)/*.cpp))
TOOL 		= $(foreach dir,$(TOOL_PATH),$(wildcard $(dir)/*.cpp))

SRC_OBJ 	= $(SRC:%.cpp=%.o)
TOOL_OBJ 	= $(TOOL:%.cpp=%.o)

TOOL_EXE 	= $(TOOL:%.cpp=%)

# 生成库
LIB_NAME = swabl-dnn
LIB = $(INIT_PATH)/lib$(LIB_NAME).so
LIBFLAGS = -shared -Wl,-soname

# 颜色
RED 		= "\e[38;5;9m"
GREEN 		= "\e[38;5;10m"
YELLOW 		= "\e[38;5;11m"
BLUE 		= "\e[38;5;12m"
PURPLE 		= "\e[38;5;13m"
WHITE 		= "\e[38;5;15m"

.PHONY: all
all: env_color $(LIB) $(TOOL_EXE)
	@echo $(GREEN)"-------------------- Make All Success --------------------"

$(TOOL_EXE): % : %.o $(LIB)
	$(GCC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -l$(LIB_NAME)

%.o: %.cpp
	$(GCC) $(CFLAGS) -o $@ -c $<

$(LIB): $(SRC_OBJ)
	@echo $(RED)"Create dynamic library, please waiting ....."
	$(GCC) $(CFLAGS) $(LIBFLAGS),$(notdir $(LIB)) -o $@ $^ $(LDFLAGS)

.PHONY: env_color
env_color:
	@echo $(RED)"Begin making, please waiting ....."$(WHITE)

.PHONY: clean
clean: env_color
	@echo $(RED)"Clean unused files, please waiting ....."$(WHITE)
	rm -f $(SRC_OBJ) $(TOOL_OBJ)
	rm -f $(TOOL_EXE)
	rm -f $(LIB)
	@echo $(GREEN)"-------------------- Make Clean Success --------------------"

.PHONY: show

show: env_color
	@echo $(GREEN)"Show all INIT-path message:"$(WHITE)
	@echo "INIT_PATH:       $(INIT_PATH)"
	@echo "=========================================="
	
	@echo $(GREEN)"Show all SRC-file message:"$(WHITE)
	@echo "SRC_PATH:        $(SRC_PATH)"
	@echo "SRC:             $(SRC)"
	@echo "SRC_OBJ:         $(SRC_OBJ)"
	@echo "=========================================="
	
	@echo $(GREEN)"Show all TOOL-file message:"$(WHITE)
	@echo "TOOL_PATH:       $(TOOL_PATH)"
	@echo "TOOL:            $(TOOL)"
	@echo "TOOL_OBJ:        $(TOOL_OBJ)"
	@echo "TOOL_EXE:        $(TOOL_EXE)"
	@echo "=========================================="
	
	@echo $(GREEN)"Show all LIB-file message:"$(WHITE)
	@echo "LIB:             $(LIB)"
	@echo "=========================================="
	@echo $(GREEN)"-------------------- Make Show Success --------------------"

# ||=====================================================================||
# || 01. LIBRARY_PATH环境变量，指定程序静态链接库文件搜索路径；          ||
# ||     LD_LIBRARY_PATH环境变量，指定程序动态链接库文件搜索路径；       ||
# || 02. gcc (-I/ -L/ -Wl,-rpath=) *.o -o exe -lxxx;                     ||
# || 03. makefile一些常见的检查规则                                      ||
# ||         (1) --just-print, 不执行参数，只打印命令，不管目标是否跟新；||
# ||         (2) --what-if=<file>, 指定一个文件，一般和"-n"一起使用，来查||
# ||             这个文件所发生的规则命令；                              ||
# || 0.4 ANSI code sequence, control color                               ||
# ||=====================================================================||
