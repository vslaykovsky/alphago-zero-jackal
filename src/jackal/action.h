#pragma once


#include "../util/utils.h"

#include <boost/functional/hash.hpp>


struct Action {

    Action() : with_items(false) {

    }

    Action(Coords coordinates_from,
           Coords coordinates_to,
           bool with_items) : coordinates_from(coordinates_from),
                              coordinates_to(coordinates_to),
                              with_items(with_items) {

    }

    Coords coordinates_from;
    Coords coordinates_to;
    int with_items;

    inline int operator==(const Action &a) const {
        return a.coordinates_from == coordinates_from and a.coordinates_to == coordinates_to and
               a.with_items == with_items;
    }

};

inline std::ostream &operator<<(std::ostream &os, const Action &a) {
    return os << "Action([" << a.coordinates_from.x << ","
              << a.coordinates_from.y << "], ["
              << a.coordinates_to.x << "," << a.coordinates_to.y << "], " << a.with_items << ")";
}

namespace std {
    template<>
    struct hash<Action> {
        std::size_t operator()(const Action &action) const {
            using boost::hash_value;
            using boost::hash_combine;
            std::size_t seed = 0;
            hash_combine(seed, hash_value(action.coordinates_from.x));
            hash_combine(seed, hash_value(action.coordinates_from.y));
            hash_combine(seed, hash_value(action.coordinates_to.x));
            hash_combine(seed, hash_value(action.coordinates_to.y));
            hash_combine(seed, hash_value(action.with_items));
            return seed;
        }
    };
}
