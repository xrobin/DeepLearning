#pragma once

#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm> // std::copy
#include <memory> // std::allocator
//#include <utility> // std::move
//using std::move;

#include "boost/numeric/conversion/cast.hpp"

/** Class shared_array_ptr
 * Handles an array pointer in a shared manner with copy counter.
 * Kind of like boost::shared_array, but here we use cleanUp to give the user the possibility to manage the object himself. 
 * By default, cleanUp is true unless the user explicitly passed an array to the constructor - in which case it is set to false and the user has to manage the pointer himself or to set cleanUp=true.
 * Use aCleanUp = false upon construction or call setCleanUp(false) later so the data itself is not deleted upon destruction, allowing to keep the pointer afterwards.
 * This is especially useful when dealing with objects from R/Rcpp, or in conjunction with .getData() so that you can then use the data however you want.
 * 
 * The operators ([], +, - and >) are not safe, i.e. they don't check that the data is in range
 */
 


template <class T, class Alloc = std::allocator<T>> class shared_array_ptr {
public:
	/** STL-like Typedefs */
	typedef typename Alloc::value_type         value_type;
	typedef typename Alloc::pointer            pointer;
	typedef typename Alloc::const_pointer      const_pointer;
	typedef typename Alloc::reference          reference;
	typedef typename Alloc::const_reference    const_reference;
	typedef typename Alloc::size_type          size_type;
	typedef typename Alloc::difference_type    difference_type;
	
private:
	size_t myDataSize; // the array size
	T* myData; // the array
	unsigned long* myCount; // how many copies?
	size_t myOffset; // do we start at index 0?
	size_t myLength; // how much of myDataSize after myOffset do we use?
	bool* cleanUp; // do we manage the object?
	
	void releaseData() {
		if (--(*myCount) == 0) {
			if (*cleanUp){
//					std::cout << "Data destructed" << std::endl;
				delete[] myData;
			}
			delete myCount;
			delete cleanUp;
//				std::cout << "Deleted pointers in shared_array_ptr" << std::endl;
		}
	}

	// constructor(shared_array_ptr&, size_t, size_t): copy with an offset and a length. Used only by operator+, operator- and operator> that check the range itself - no check is done here
	// Note that anOffset is the new offset, not its increment, and must be valid!
	shared_array_ptr<T, Alloc>(const shared_array_ptr& oldWeights, size_t anOffset, size_t aLength) : myDataSize(oldWeights.myDataSize), myData(oldWeights.myData),
		myCount(oldWeights.myCount), myOffset(anOffset), myLength(aLength), cleanUp(oldWeights.cleanUp) {
		(*myCount)++;
//						std::cout << "Copied shared_array_ptr with offset. Count = " << *myCount << ", offset = " << myOffset << ", length = " << myLength << ", pointer = " << myData << std::endl;
		}

public:
	
	/** constructor(size_t, bool): will allocate the array itself */
	explicit shared_array_ptr<T, Alloc>(const size_t aDataSize, const bool aCleanUp = true) : myDataSize(aDataSize), myData(new T[myDataSize]), myCount(new unsigned long(1)), myOffset(0), myLength(myDataSize), cleanUp(new bool(aCleanUp)) {
//			std::cout << "Created shared_array_ptr from aDataSize (size_t). Count = " << *myCount << ", offset = " << myOffset << ", length= " << myLength << ", pointer = " << myData << std::endl;
	}
	/** constructor(long|int, bool): in addition, throws boost::bad_numeric_cast, boost::positive_overflow or boost::negative_overflow if aDataSize cannot be represented in a size_t. */
	explicit shared_array_ptr<T, Alloc>(const long aDataSize, const bool aCleanUp = true) : myDataSize(aDataSize), myData(new T[myDataSize]), myCount(new unsigned long(1)), myOffset(0), myLength(myDataSize), cleanUp(new bool(aCleanUp)) {
//			std::cout << "Created shared_array_ptr from aDataSize (long). Count = " << *myCount << ", offset = " << myOffset << ", length= " << myLength << ", pointer = " << myData << std::endl;
	}
	
	/** The constructor from aDataSize will create the data (initialized to 0) */
	explicit shared_array_ptr<T, Alloc>(const int aDataSize, const bool aCleanUp = true) : myDataSize(boost::numeric_cast<std::size_t>(aDataSize)), myData(new T[myDataSize]), myCount(new unsigned long(1)), myOffset(0), myLength(myDataSize), cleanUp(new bool(aCleanUp)) {
//			std::cout << "Created shared_array_ptr from aDataSize (int). Count = " << *myCount << ", offset = " << myOffset << ", length= " << myLength << ", pointer = " << myData << std::endl;
	}
	
	/** constructor with begin/end iterators from STL or other standard containers. Makes a copy of the data.
	InputIt must meet the requirements of InputIterator <http://en.cppreference.com/w/cpp/concept/InputIterator>. 
	*/
	template< class InputIt > shared_array_ptr<T, Alloc>(InputIt first, InputIt last, const bool aCleanUp = true) : myDataSize(std::distance(first, last)), myData(new T[myDataSize]), myCount(new unsigned long(1)), myOffset(0), myLength(myDataSize), cleanUp(new bool(aCleanUp)) {
		for (size_t i = 0; i < myDataSize; ++i) {
			myData[i] = *first++;
		}
//			std::cout << "Created shared_array_ptr from two iterators. Count = " << *myCount << ", offset = " << myOffset << ", length= " << myLength << ", pointer = " << myData << std::endl << std::flush;
		
	}
	/** The constructor from a Container will make a copy of the data
	The container must provide .size(), .begin() and .end() methods. See <http://en.cppreference.com/w/cpp/concept/Container>
	Also it must point to elements of type T.
	*/
	explicit shared_array_ptr<T, Alloc>(const std::vector<T, Alloc>& aData, const bool aCleanUp = true) : myDataSize(aData.size()), myData(new T[aData.size()]), myCount(new unsigned long(1)), myOffset(0), myLength(myDataSize), cleanUp(new bool(aCleanUp)) {
//			std::cout << "Created shared_array_ptr from a vector. Count = " << *myCount << ", offset = " << myOffset << ", length= " << myLength << ", pointer = " << myData << std::endl;
		// aData is going to disappear at some point. We need to copy the data before it does so.
			std::copy(aData.begin(), aData.end(), myData);
	}
	/** constructor(T*, size_t, bool): using an array already allocated elsewhere. DOES NOT COPY THE DATA, operates on it directly.
	This is why aCleanUp = false by default: it is expected that the user takes care of it.
	Of course you can turn aCleanUp to true so that shared_array_ptr takes care of cleaning up the mess. */
	shared_array_ptr<T, Alloc>(T* aData, const size_t aDataSize, const bool aCleanUp = false) : myDataSize(aDataSize), myData(aData), myCount(new unsigned long(1)), myOffset(0), myLength(aDataSize), cleanUp(new bool(aCleanUp)) {
//			std::cout << "Created shared_array_ptr from *aData. Count = " << *myCount << ", offset = " << myOffset << ", length= " << myLength << ", pointer = " << myData << std::endl;
	}
	// 1. Copy constructor
	shared_array_ptr<T, Alloc>(const shared_array_ptr<T>& old_ptr) : myDataSize(old_ptr.myDataSize), myData(old_ptr.myData), myCount(old_ptr.myCount), myOffset(old_ptr.myOffset), myLength(old_ptr.myLength), cleanUp(old_ptr.cleanUp) {
		(*myCount)++;
//			std::cout << "Copied shared_array_ptr. Count = " << *myCount << ", offset = " << myOffset << ", length= " << myLength << ", pointer = " << myData << std::endl;
	}
	
	// Move constructor
	// No move constructor here - we have a shared pointer, not a unique pointer, and moving it doesn't make much sense.
	/*shared_array_ptr<T>(shared_array_ptr<T>&& old_ptr) : myDataSize(move(old_ptr.myDataSize)), myData(move(old_ptr.myData)), myCount(move(old_ptr.myCount)), myOffset(move(old_ptr.myOffset)), myLength(move(old_ptr.myLength)), cleanUp(move(old_ptr.cleanUp)) {
		//(*myCount)++;
		std::cout << "Moved shared_array_ptr. Count = " << *myCount << ", offset = " << myOffset << ", length= " << myLength << ", pointer = " << myData << std::endl;
	}*/

	// 2. Copy assign constructor
	shared_array_ptr<T, Alloc>& operator=(const shared_array_ptr<T>& rhs) {
		if (this != &rhs) {
			releaseData();
			myData = rhs.myData;
			myCount = rhs.myCount;
			(*myCount)++;
			myDataSize = rhs.myDataSize;
			myLength = rhs.myLength;
			cleanUp = rhs.cleanUp;
			myOffset = rhs.myOffset;
		}
//			std::cout << "Assigned shared_array_ptr. Count = " << *myCount << ", offset = " << myOffset << ", length= " << myLength << ", pointer = " << myData << std::endl;
		return *this;
	}
	// 3. Delete
	~shared_array_ptr<T, Alloc>() {releaseData();}
	
	T& at(const size_t anIndex) const {
		if (anIndex >= myLength)  throw std::out_of_range("Offset out of range!");
		return myData[myOffset + anIndex];
	}
	
	T& operator[] (const size_t anIndex) const {
		return myData[myOffset + anIndex];
	}
	
	/** Comparison: is it the same object?*/
	bool operator==(const shared_array_ptr<T>& rhs) {
		return this == &rhs;
	}
	
	/** Comparison: is it the same object?*/
	bool operator!=(const shared_array_ptr<T>& rhs) {
		return this != &rhs;
	}

	// Pointer addition operator. Adds to the current offset, i.e. can be negative to go back in the array, and reduce length accordingly
	shared_array_ptr<T, Alloc> operator+ (const std::ptrdiff_t& anOffset) const {
		//if ( (std::intptr_t)myOffset + anOffset < 0 || (std::intptr_t)myOffset + anOffset >= myDataSize || (std::intptr_t)myLength - anOffset <= 0 ) throw std::out_of_range("Offset out of range!");
		return shared_array_ptr<T>(*this, myOffset + anOffset, myLength - anOffset);
	}
	// Pointer substraction operator. Reduces length of anEndOffset, i.e. can be negative to go back in the array, and increase length accordingly
	shared_array_ptr<T, Alloc> operator- (const std::ptrdiff_t& anEndOffset) const {
		//if ( (std::intptr_t)myLength - anEndOffset <= 0 || (std::intptr_t)myOffset + myLength - anEndOffset > myDataSize) throw std::out_of_range("Offset out of range!");
		return shared_array_ptr<T>(*this, myOffset, myLength - anEndOffset);
	}
	// Pointer length operator. Sets length to aLength
	shared_array_ptr<T, Alloc> operator> (const size_t& aLength) const {
		//if (myOffset + aLength > myDataSize) throw std::out_of_range("Offset out of range!");
		return shared_array_ptr<T>(*this, myOffset, aLength);
	}
	// Output operator
	friend std::ostream& operator<<(std::ostream& os, const shared_array_ptr<T>& ptr) {
		os << ptr[0];
		for (size_t i = 1; i < ptr.size(); i++) {
			os << ", " << ptr[i];
		}
		return os;
	}

	// Getters
	T* data() const {return myData;} // Warning: returns the whole data, without offset, otherwise it's a mess to delete[] the ptr later on...
	T* getOffsetData() const {return myData + myOffset;} // Warning: returns the data minus the offset, so don't try to delete[] this later on...
	unsigned long* count() const {return myCount;}
	size_t offset() const {return myOffset;}
	size_t totalSize() const {return myDataSize;}
	size_t size() const {return myLength;}
	bool* getCleanUp() const {return cleanUp;}
	// Setters
	void setCleanUp(const bool aCleanUp = true) {delete cleanUp; cleanUp = new bool(aCleanUp);} // sets on all copies of this object - so the last one actually perform what is needed
	// Cast object to T* (= basically get the pointer)
	//explicit operator T*() {return getOffsetData();}
	
	T* begin() {
		return getOffsetData();
	}
	T* end() {
		return getOffsetData() + size();
	}
	const T* begin() const {
		return getOffsetData();
	}
	const T* end() const {
		return getOffsetData() + size();
	}
	
	// Clone
	/** Clone the whole array from myOffset */
	shared_array_ptr<T, Alloc> clone() { 
		return(clone(myLength));
	}

	/** Clone the data only into a vector */
	std::vector<T, Alloc> toVector() const {
		std::vector<T, Alloc>  v(getOffsetData(), getOffsetData() + myLength);
		return v;
	}

private:
	/** Clone only the first cloneLength elements of the array following myOffset */
	shared_array_ptr<T, Alloc> clone(size_t cloneLength) { // return a deep copy of the object - but without the offset
		T* newData = new T[cloneLength];
		/*size_t i = 0; size_t j = myOffset;
		while (i < cloneLength) {
			newData[i] = myData[j];
			++i; ++j;
		}*/
		size_t i = 0;
		for (T& element: *this) {
			newData[i] = element;
			++i;
		}
		return shared_array_ptr<T, Alloc>(newData, cloneLength, true);
	}
};