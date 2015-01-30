#include <iostream>
using namespace std;

void addNumbers(int *Array,int n);

void merge(int a[], const int low, const int mid, const int high)
{
	// Variables declaration.
	int * b = new int[high+1-low];
	int h,i,j,k;
	h=low;
	i=0;
	j=mid;
	// Merges the two array's into b[] until the first one is finish
	while((h<mid)&&(j<high))
	{
		if(a[h]<=a[j])
		{
			b[i]=a[h];
			h++;
		}
		else
		{
			b[i]=a[j];
			j++;
		}
		i++;
	}
	// Completes the array filling in it the missing values
	while(h<mid)
	{
		b[i]=a[h];
		h++;
		i++;
	}
	while(j<high)
	{
		b[i]=a[j];
		j++;
		i++;
	}
	// Prints into the original array
	for(k=0;k<high-low;k++)
	{
		a[k+low]=b[k];
	}
	delete[] b;
}

void merge_sort(int a[], const int low, const int high) // Recursive sort ...
{
	int mid;
	if(low+1<high)
	{
		mid=(low+high)/2;
		merge_sort(a, low,mid);
		merge_sort(a, mid,high);
		merge(a, low,mid,high);
	}
}
void display_array(int size, int* intArray)  //array displaying function
{
	cout << "\n\nDisplaying Array"<<endl;
	cout << "Array Size: " << size<<endl << "Elements: ";
	for(int i=0; i<size; i++)
	{
		cout << intArray[i];
		if(i < size-1)
			cout <<", ";
		else
			cout <<"\n\n";
	}
}
int main(char argc, int * argv[])
{
	int *a=new int[10];
	addNumbers(a,10);
	display_array(10,a);
	//
	int arraySize=10;
	// a[] is the array to be sorted. ArraySize is the size of a[] ...
	merge_sort(a, 0, 10 );        // would be more natural to use
	//merge_sort(a, 0, arraySize ); // so please try ;-)
	display_array(10,a);
	system("PAUSE");
	// some work
	return 0;
}
void addNumbers(int *Array,int n) //numbers generating function
{
	for(int i=0; i<n; i++)
	{
		Array[i]= (rand() %100)+1;
	}
}