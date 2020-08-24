import copy
import hashlib
import json

class HCubeSet:
    '''
    Represents a set of hypercubes that are associated with a specific sensitive
    attribute values/constraints.
    '''
    def __init__(self, sensitive_attrs, hcubes=None):
        '''
        sensitive_attrs is a map: string -> Constraint
          maps sensitive attribute name (string) to a Constraint
          instance for this attribute.
        '''
        self.sensitive_attrs = sensitive_attrs

        # self._hcubes is a map: hid -> hcube
        # from hypercube id strings to HCube instances
        # the hid are specific to a scikit decision tree that the hcube was derived from
        # these hid's are useful for matching hcubes across HCubeSet instances
        if hcubes is None:
            hcubes = {}
        self._hcubes = hcubes

    @staticmethod
    def crossproduct(hsets1, hsets2):
        '''
        Given two maps:
        hsets1: str -> HCubeSet instance
        hsets2: str -> HCubeSet instance
        
        Computes a cross product of the two maps, returns:
        hsetsx: str -> HCubeSet instance
        Where hsetsx[i] corresponds to product(hsets1[k],hsets1[l]) for some k,l
        '''
        hsetsx = {}
        for k1, hset1 in hsets1.items():
            for k2, hset2 in hsets2.items():
                hsetx = HCubeSet.product(hset1, hset2)
                kx = k1 + " X " + k2
                hsetsx[kx] = hsetx
        return hsetsx

    @staticmethod
    def product(hset1, hset2):
        '''Computes a new HCubeSet instance that is a merger between HCubeSet
        instances hset1 and hset2.  Returns a newly created HCubeSet
        instance.
        The returned HCubeSet instance satisfies two properties:
        1. it contains just the common hcubes from hset1 and hset2
        2. it contains the union of the sensitive attribute constraints from hset1 and hset2
        '''
        hcubes1 = hset1.get_hcubes_map()
        hcubes2 = hset2.get_hcubes_map()
        
        # Determine the set of hcubes hids in both hsets
        hids1 = hcubes1.keys()
        hids2 = hcubes2.keys()
        hids_common = hids1 & hids2

        # Create hcubes_common based on hids_common
        hcubes_common = {} 
        for hid in hids_common:
            hcubes_common[hid] = copy.deepcopy(hcubes1[hid])

        # Now, compute the union of the sensitive attributes from hset1 and hset2
        sens1 = hset1.sensitive_attrs
        sens2 = hset2.sensitive_attrs

        sens_union = copy.deepcopy(sens1)
        sens_union.update(sens2)
        
        # Create a new HCubeSet that:
        # 1. contains just the common hcubes
        # 2. contains the union of the sensitive attribute constraints
        hsetx = HCubeSet(sens_union, hcubes_common)
        return hsetx

    def add_hcube(self, hcube):
        '''
        Adds a hypercube to this set of hypercubes.
        '''
        assert (hcube.hid not in self._hcubes)
        self._hcubes[hcube.hid] = hcube
        return

    def get_hcubes_map(self):
        return self._hcubes

    def get_hcubes_list(self):
        '''
        Returns a list of underlying HCube instances
        '''
        return self._hcubes.values()

    def add_point(self, point, point_hid):
        '''If the point satisfies sensitive constraints and maps to some
        hcube based on the hid, then adds the point to that hcube.
        Returns True if the point was added to an hcube; False otherwise.
        '''
        # Check if point does not satisfy sensitive constraints.
        for fname, fname_c in self.sensitive_attrs.items():
            # if not fname in point:
            #     raise RuntimeError("No value for feature '" + fname + "' in point: " + str(point))
            if not fname_c.satisfies(point[fname]):
                return False
        
        if not point_hid in self._hcubes:
            # This HCubeSet does not contain a matching hcube for this point.
            return False
        
        self._hcubes[point_hid].add_point(point)
        return True

    def get_sensitive_attrs(self):
        return self.sensitive_attrs

    def get_hcubes_list_copy(self):
        '''
        Returns a list of HCube instances that are:
        1. copies of what hcubes.values() contains
        2. include all the sensitive_attr constraints
        '''
        hc_list = []
        for hcube in self._hcubes.values():
            hcube_cp = copy.deepcopy(hcube)
            for attr_c in self.sensitive_attrs.items():
                hcube_cp.add_constraint(attr_c)
            hc_list.append(hcube_cp)
        return hc_list

    def __str__(self):
        s = "HCubeSet[Sensitive attrs: "
        for at in self.sensitive_attrs:
            s += str(at) + ", "
        s+= "]\n"

        counter = 0
        for hc in self._hcubes:
            s+= str(hc) + ", "
            counter += 1
            if counter == 2:
                s+= str(hc) + "... <" + str(len(self._hcubes) - 5) + " More>"
                break
        s+= "]"
        return s

    def __repr__(self):
        return self.__str__()


class Constraint:
    def __init__(self, name, upperBound, thresh):
        '''
        name: the name of the feature/variable that is being constrained
        upperBound is True: constraint is an upper bound (i.e., <=)
        upperBound is False: constraint is a lower bound (i.e., >)
        thresh: a double value/threshold
        '''
        self.name = name
        self.upperBound = upperBound
        self.thresh = thresh

    def __str__(self):
        if self.upperBound:
            return "(" + self.name + " <= " + str(self.thresh) + ")"
        else:
            return "(" + self.name + " > " + str(self.thresh) + ")"

    def __repr__(self):
        return self.__str__()

    def get_name(self):
        return self.name

    def is_upper(self):
        return self.upperBound

    def is_lower(self):
        return (not self.upperBound)

    def isStronger(self,c):
        return False

    def satisfies(self, val):
        '''
        Whether or not val satisfies this constraint.
        '''
        if self.upperBound:
            # The constraint is an upper bound (i.e., <=)
            if not (val <= self.thresh):
                return False
        else:
            # The constraint is a lower bound (i.e., >)
            if not (val > self.thresh):
                return False
        return True


class HCube:
    '''
    Represents a hypercube, which is a set of constraints on features,
    and a value for the classification of points that fall into the
    hypercube.
    
    Assumption is that if the feature has no constraints, then the
    hypercube accepts any values for this feature
    '''
    def __init__(self,constraints=None,val=None,hid=None,pts=None,desc=None,dataList=None):
        '''
        map: str (feature) -> list of [X,constraint] pairs
        This defines the hypercube as constraints on the feature/dimensions of the dataset.
        Note that a feature may be constrained multiple times.
        if X is True then the constraint is an upper bound (i.e., <=)
        if X is False then the constraint is a lower bound (i.e., >)
        constraint is simply a double value/threshold
        
        hid is the hypercube-id (string) that uniquely identifies this hypercube in the tree
        where the hypercube originated.
        '''
        if constraints is None:
            constraints = {}
        if pts is None:
            pts = set()
        if desc is None:
            desc = []
        self.constraints = constraints
        self.val = val
        self.hid = hid
        self.pts=pts
        self.desc = desc
        self.passing_rate = 0
        for point in self.pts:
            self.passing_rate += dataList[point]['frequency']

    def __str__(self):
        ret = "HCube[Value: " + str(self.val) + ", Constraints: \n"
        for cname,clist in self.constraints.items():
            ret += "\t" + str(cname) + ": " + str(clist) + "\n"
        ret += "]"
        return ret

    def __repr__(self):
        return self.__str__()

    def add_constraint(self,constraint):
        name = constraint.get_name()
        if self.constraints.get(name) == None:
            self.constraints[name] = [constraint]
        else:
            self.constraints[name].append(constraint)

    def rm_constraint(self,name,constraint):
        self.constraints[name].remove(constraint)

    def rm_constraints(self,name):
        # remove self.constraints[name]
        self.constraints.pop(name, None)

    def rm_all_constraints(self):
        self.constraints = {}

    def get_constraints(self,name=None):
        '''
        Returns all constraints if name is None
        If hcube not constrained on feature with name, then returns []
        '''
        if name is None:
            # Get all constraints.
            return self.constraints
        if self.constraints.get(name) == None:
            return []
        return self.constraints[name]

    def set_value(self,val):
        self.val = val

    def get_value(self):
        return self.val

    def get_hid(self):
        return self.hid

    def add_point(self,point):
        # Do not allow duplicate points
        # This check is too expensive for large datasets: 
        # assert (point not in self.pts)
        # before = len(self.pts)
        self.pts.add(point['index'])
        # assert(before == len(self.pts) - 1)
        self.passing_rate += point['frequency']
        return
    
    def remove_point(self,point):
        # before = len(self.pts)
        self.pts.remove(point['index'])
        # assert(before == len(self.pts) + 1)
        self.passing_rate -= point['frequency']
        return

    # def print_points(self):
    #     Property = ['Property_A121', 'Property_A122', 'Property_A123', 'Property_A124']
    #     for point in self.pts:
    #         print("\t" + hashlib.sha1(json.dumps(point, sort_keys=True).encode()).hexdigest())
    #         for p in Property:
    #             print("\t" + p + ": " + str(point[p]))
            
    def get_points(self):            
        return self.pts

    def add_desc(self,desc_):
        self.desc.append(desc_)
        return

    def get_desc(self):
        return self.desc
    
    def get_passing_rate(self):
        '''
        Returns the number of points associated with this hypercube.
        '''
        return self.passing_rate

    def has_constraint(self, name):
        if self.constraints.get(name) == None:
            return False
        return True
    
    def intersect(self,hcube):
        return self

    def refine_one_hot(self, attr_to_refine,dataList):
        '''
        Invoked on an HCube instance.
        1. Changes the current HCube to include a lower bound on attr_to_refine
        2. Returns a new HCube that is created based off self, and includes an upper bound constraint on attr_to_refine
        The points in the two hcubes will be the union of self.pts
        Each hcube's points will satisfy it's respective constraints.
        '''
        c_upper = Constraint(attr_to_refine, True, 0.5)
        c_lower = Constraint(attr_to_refine, False, 0.5)

        h_new_constraints = copy.deepcopy(self.constraints)
        # The hid of h_new *must* be unique. We achieve this by:
        # 1. label it as a refinement with 'R'
        # 2. Encode the attr_to_refine in the hid 
        # NOTE: This assume that the attr_to_refine can only be refined on ONCE
        h_new_hid = self.hid + 'L[' + attr_to_refine + ']'
        h_new = HCube(constraints=h_new_constraints, val=self.val, hid=h_new_hid)
        h_new.add_constraint(c_upper)
        self.add_constraint(c_lower)

        # Note: operate on copy of points for safety
        points_copy = copy.deepcopy(self.pts)
        for point in points_copy:
            if not c_lower.satisfies(dataList[point][attr_to_refine]):
                # Add this point into the new hypercube since it
                # doesn't satisfy the lower bound constraints now
                # associated with self (this hypercube)
                h_new.add_point(dataList[point])
                # Remove this point from this hypercube
                self.remove_point(dataList[point])
        return h_new

    def refine_cont_one_hot(self,attr_to_refine,thresh,dataList):
        '''
        Invoked on an HCube instance.
        1. Changes the current HCube to include a lower bound on attr_to_refine
        2. Returns a new HCube that is created based off self, and includes an upper bound constraint on attr_to_refine
        The points in the two hcubes will be the union of self.pts
        Each hcube's points will satisfy it's respective constraints.
        '''
        c_upper = Constraint(attr_to_refine, True, thresh)
        c_lower = Constraint(attr_to_refine, False, thresh)

        h_new_constraints = copy.deepcopy(self.constraints)
        # The hid of h_new *must* be unique. We achieve this by:
        # 1. label it as a refinement with 'R'
        # 2. Encode the attr_to_refine in the hid 
        # NOTE: This assume that the attr_to_refine can only be refined on ONCE
        h_new_hid = self.hid + 'L[' + attr_to_refine + ']'
        h_new = HCube(constraints=h_new_constraints, val=self.val, hid=h_new_hid)
        h_new.add_constraint(c_upper)
        self.add_constraint(c_lower)

        # Note: operate on copy of points for safety
        points_copy = copy.deepcopy(self.pts)
        for point in points_copy:
            if not c_lower.satisfies(dataList[point][attr_to_refine]):
                # Add this point into the new hypercube since it
                # doesn't satisfy the lower bound constraints now
                # associated with self (this hypercube)
                h_new.add_point(dataList[point])
                # Remove this point from this hypercube
                self.remove_point(dataList[point])
        return h_new

    def one_hot_refineables(self, related_features):
        '''
        Input:
        - related_features is a set of the one-hot encoded values of some base feature
        (e.g., ['Sex_A91', 'Sex_A92', 'Sex_A93', 'Sex_A94'] are one-hot encodings of 'Sex')
        If the base feature can be further refined, this functions returns
        the discrete list of one-hot encoded dimensions along which it can be refined:
        
        Example 1: if hcube has the constraint 'Sex_A91 => 0.5', then the possible set of
        refinements is the empty list: [] (i.e., Sex can only be Sex_A91) 
        
        Example 2: if hcube has two constraints: 'Sex_A91 < 0.5' and 'Sex_A92 < 0.5', then
        the possible set of refinements is: ['Sex_A93', 'Sex_A94'].
        
        Example 3: if there are no constraints on any related_features, then
        the returned list is all the related_features.
        '''
        ret = copy.deepcopy(related_features)
        for f in related_features:
            # print(f)
            f_consts = self.constraints.get(f)
            # print(f_consts)
            if f_consts == None:
                continue
            for c in f_consts:
                if c.is_lower():
                    # This hcube is completely refined since this (f) attribute is set
                    return []
                if c.is_upper():
                    # The hcube has a constraint that eliminates this (f) attribute
                    # ret should exclude f
                    ret.remove(f) 
        return ret    

    def inside_hypercube(self, features):
        '''
        features is a map: str(feature) -> value
        features describe a point in R^n
        '''
        #print("inside hypercube: features = ", features)
        #print(self.constraints)
        for (feature,val) in features.items():
            # skip if the hypercube does not constraint this feature
            if feature not in self.constraints:
                continue
            for constraint in self.constraints[feature]:
                assert(constraint.get_name() == feature)
                if not constraint.satisfies(val):
                    return False
        return True
