# $UWHPSC/codes/fortran/multifile1/Makefile
#THIS IS OUR TARGET, ALWAYS COMES IN THE LEFT
#FIRST FINAL TARGET: SECOND DEPENDENCIES
#- Tab -- command    
OBJECTS = routines.o main.o
.PHONY: clean


output.txt: main.out
#	./main.out

main.out: $(OBJECTS)
	gfortran $(OBJECTS) -o main.out

%.o : %.f95
	gfortran -c $<

clean:
	rm -f $(OBJECTS)
